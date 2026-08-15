"""Falsification suite for the v4 campaign status record.

The record under test -- ``artifacts/animal-planet/evaluation/v4-campaign-status.json``
and its ``.md`` companion -- is a *claim*, and this module is the only thing
standing between that claim and wishful reporting.  Nothing here trusts the
record: every number it states is re-derived from the frozen protocol, from the
final report, or from bytes on disk, and every absence it asserts is settled by
an actual scan.

Two failure modes drive the design.

**Zero-as-a-measurement.**  A campaign that has executed nothing has no
readings.  Reporting a floor count of ``0`` would not be humility, it would be
a fabricated measurement -- indistinguishable in a table from a real count of
zero qualifying families.  So the record splits *action counts* (0, because the
action provably never happened) from *observations* (the sentinel string, because
nothing was ever measured), and this suite walks the whole document to prove no
floor, readiness, or gap quantity ever appears as a number outside the block of
preregistered thresholds copied verbatim from the frozen plan.

**A stale "unstarted" claim.**  ``tests/test_confirmatory_holdout_v4_packet.py``
deliberately asserts no blanket absence of ``segments/``, ``probes/``,
``manifest.json`` or ``seal-receipt.json`` inside the frozen namespace, because
the downstream accrue-and-seal sibling legitimately creates them.  This suite is
the one place that *does* assert that absence, and that is the point: the record
claims the campaign has executed zero slots, so the moment real accrual
artifacts appear, this record has become false and must be rewritten rather than
silently kept.  A red test here is the correct outcome, not a fragile one.

This module executes no slot, arms no cadence unit, opens no source, and writes
nothing outside pytest's own temporary space.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "animal-planet" / "evaluation"
NAMESPACE = EVAL / "confirmatory-holdout-v4"
RECORD_JSON = EVAL / "v4-campaign-status.json"
RECORD_MD = EVAL / "v4-campaign-status.md"
PLAN_PATH = NAMESPACE / "analysis-plan.json"
FINAL_REPORT_PATH = EVAL / "final-report.json"
RUNBOOK_PATH = ROOT / "docs" / "v4-campaign-runbook.md"

SENTINEL = "not observed"

# A ledger entry is ``NNNNNNNN.json``; see ``ledger_entry_filename`` in
# ``scripts/ap_confirmatory_accrual_v4.py``.
LEDGER_ENTRY_RE = re.compile(r"^\d{8}\.json$")

# Directories a repository scan must not descend into: VCS internals, caches,
# and nested worktrees that carry their own copy of the tree.
PRUNED_DIRECTORY_NAMES = frozenset(
    {".git", ".worktrees", ".cache", "__pycache__", ".pytest_cache", ".venv", "node_modules"}
)

# ``floor_evaluations_performed`` is the single numeric key outside the
# threshold block whose name contains "floor".  It counts an *action* the
# campaign never took, not a measured quantity, so it is allowlisted by exact
# path rather than by pattern.
ACTION_COUNT_FLOOR_KEY_PATH = ("campaign_execution", "floor_evaluations_performed")

THRESHOLD_BLOCK = "preregistered_floor_thresholds"
OBSERVATION_BLOCK = "campaign_observations"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record() -> dict[str, Any]:
    return json.loads(RECORD_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def markdown() -> str:
    return RECORD_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def plan() -> dict[str, Any]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def final_report() -> dict[str, Any]:
    return json.loads(FINAL_REPORT_PATH.read_text(encoding="utf-8"))


def walk(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield ``(key path, leaf value)`` for every leaf in the document."""

    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from walk(value, path + (f"[{index}]",))
    else:
        yield path, node


def scan_for_campaign_artifacts() -> dict[str, list[str]]:
    """Every in-repository artifact whose existence would mean the campaign ran.

    Accrual ledgers live in the operator's ``--work-dir`` outside this
    repository, so their absence here is expected and is checked by shape
    (a directory holding ``NNNNNNNN.json``) rather than by location.  The
    namespace artifacts, by contrast, are created *inside* the repository by a
    real accrual or seal, and are the load-bearing in-repo falsifier.
    """

    found: dict[str, list[str]] = {"ledger_entries": [], "namespace_accrual": []}

    for current, directories, files in os.walk(ROOT):
        directories[:] = [d for d in directories if d not in PRUNED_DIRECTORY_NAMES]
        here = Path(current)
        for name in files:
            if LEDGER_ENTRY_RE.match(name):
                found["ledger_entries"].append(str((here / name).relative_to(ROOT)))

    for name in ("segments", "probes", "manifest.json", "seal-receipt.json"):
        candidate = NAMESPACE / name
        if candidate.exists():
            found["namespace_accrual"].append(str(candidate.relative_to(ROOT)))

    return found


def sha256_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The record exists, parses, and sits outside the frozen namespace
# ---------------------------------------------------------------------------


def test_record_files_exist_outside_the_frozen_namespace() -> None:
    for path in (RECORD_JSON, RECORD_MD):
        assert path.is_file(), f"{path} is missing"
        assert NAMESPACE not in path.parents, (
            f"{path} resolves inside the byte-locked namespace; the record must live outside it"
        )


def test_record_is_canonical_utf8_json_without_duplicate_keys() -> None:
    raw = RECORD_JSON.read_bytes()
    raw.decode("utf-8")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        for key, _ in pairs:
            assert key not in seen, f"duplicate key {key!r} in the record"
            seen.add(key)
        return dict(pairs)

    json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)


def test_record_declares_itself_aggregate_only_and_result_free(record: dict[str, Any]) -> None:
    contract = record["record_contract"]
    assert contract["aggregate_only"] is True
    assert contract["is_not_a_confirmatory_result"] is True
    assert contract["states_no_floor_result"] is True
    assert contract["states_no_readiness_status"] is True
    assert contract["states_no_packet_seal"] is True
    assert contract["lives_outside_frozen_namespace"] is True
    assert contract["unobserved_quantities_never_inferred_estimated_or_fabricated"] is True
    assert record["privacy"]["aggregate_only"] is True
    assert record["privacy"]["case_level_output"] is False
    assert record["privacy"]["source_named"] is False


# ---------------------------------------------------------------------------
# Falsifier 1: no floor-count field is presented as a measurement
# ---------------------------------------------------------------------------


def measurement_key_names(plan: dict[str, Any]) -> frozenset[str]:
    readiness = plan["readiness"]
    names = set(readiness["holdout_floor"]) | set(readiness["shadow_floor"])
    names |= {
        "readiness_status",
        "probe_receipt_status",
        "floor_gap",
        "snapshot_set_digest",
        "segment_identity",
        "packet_seal_state",
        "packet_manifest_sha256",
        "active_services_tuple",
        "source_binding_core",
    }
    return frozenset(names)


def test_every_observation_leaf_is_the_sentinel_string(record: dict[str, Any]) -> None:
    """Not one of them may be a number -- least of all zero."""

    assert record["not_observed_sentinel"] == SENTINEL
    assert record["campaign_observations_are_all_unobserved"] is True

    leaves = list(walk(record[OBSERVATION_BLOCK]))
    assert leaves, "the observation block is empty"
    for path, value in leaves:
        where = ".".join(path)
        assert value == SENTINEL, (
            f"{OBSERVATION_BLOCK}.{where} is {value!r}; every unobserved quantity must be "
            f"exactly {SENTINEL!r}"
        )
        assert not isinstance(value, (int, float, bool)), (
            f"{OBSERVATION_BLOCK}.{where} carries a numeric reading for a quantity "
            "the campaign never measured"
        )


def is_floor_shaped_key(leaf_key: str, guarded: frozenset[str]) -> bool:
    """Whether a key names a floor/readiness quantity, however it is dressed up.

    Exact-match alone is too weak: a fabricated count smuggled in as
    ``observed_organic_events_min`` would slip past a set membership test while
    reading, in any table, exactly like a measurement.  So the check is on
    shape -- every frozen floor key ends in ``_min``, and the guarded names are
    matched as substrings rather than whole keys.
    """

    lowered = leaf_key.lower()
    if any(name.lower() in lowered for name in guarded):
        return True
    return lowered.endswith("_min") or "floor" in lowered or "readiness" in lowered


def test_no_floor_or_readiness_quantity_appears_as_a_number(
    record: dict[str, Any], plan: dict[str, Any]
) -> None:
    """The whole document, not just the observation block, is swept."""

    guarded = measurement_key_names(plan)
    offenders: list[str] = []
    for path, value in walk(record):
        if not path:
            continue
        if path[0] == THRESHOLD_BLOCK:
            continue  # preregistered requirements, checked verbatim below
        if path == ACTION_COUNT_FLOOR_KEY_PATH:
            continue  # an action count, asserted to be 0 elsewhere
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if is_floor_shaped_key(path[-1], guarded):
            offenders.append(f"{'.'.join(path)} = {value!r}")
    assert not offenders, (
        "these fields present an unobserved floor/readiness quantity as a measurement: "
        + "; ".join(sorted(set(offenders)))
    )


@pytest.mark.parametrize(
    "fabricated_key",
    (
        "organic_events_min",
        "observed_organic_events_min",
        "holdout_floor_organic_events",
        "measured_readiness_events",
        "unseen_in_dev_repeated_automatic_events_min_observed",
    ),
)
def test_the_sweep_catches_a_fabricated_count_under_any_name(
    record: dict[str, Any], plan: dict[str, Any], fabricated_key: str
) -> None:
    """The guard is proven to have teeth, not merely to be satisfied."""

    guarded = measurement_key_names(plan)
    assert is_floor_shaped_key(fabricated_key, guarded), (
        f"a fabricated count named {fabricated_key!r} would pass the sweep"
    )
    # ...and no key the record legitimately carries trips it.
    for path, value in walk(record):
        if path and path[0] != THRESHOLD_BLOCK and path != ACTION_COUNT_FLOOR_KEY_PATH:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert not is_floor_shaped_key(path[-1], guarded)


def test_observation_keys_cover_every_frozen_floor_key(
    record: dict[str, Any], plan: dict[str, Any]
) -> None:
    """A floor key cannot be quietly dropped rather than reported as unobserved."""

    readiness = plan["readiness"]
    observations = record[OBSERVATION_BLOCK]
    for group in ("holdout_floor", "shadow_floor"):
        assert set(observations[group]) == set(readiness[group]), (
            f"{OBSERVATION_BLOCK}.{group} does not mirror the frozen {group} key set"
        )
    assert observations["readiness_status"] == SENTINEL
    assert observations["probe_receipt_status"] == SENTINEL
    assert observations["floor_gap"] == SENTINEL


def test_thresholds_are_copied_verbatim_from_the_frozen_plan(
    record: dict[str, Any], plan: dict[str, Any]
) -> None:
    """Requirements may be reproduced; they may not drift or be invented."""

    block = record[THRESHOLD_BLOCK]
    readiness = plan["readiness"]
    for group in ("holdout_floor", "shadow_floor"):
        assert block[group] == readiness[group], f"{THRESHOLD_BLOCK}.{group} drifted from the plan"
    assert block["initial_status"] == readiness["initial_status"] == "unprobed"
    assert block["source_sha256"] == sha256_of(PLAN_PATH)
    assert "never measurements" in block["role"]


# ---------------------------------------------------------------------------
# Falsifier 2: the claimed ledger state matches disk
# ---------------------------------------------------------------------------


def test_claimed_ledger_state_matches_the_actual_absence_on_disk(
    record: dict[str, Any],
) -> None:
    observed = scan_for_campaign_artifacts()
    ledger_present_on_disk = bool(observed["ledger_entries"])

    ledger = record["campaign_execution"]["ledger"]
    assert ledger["claim"] == "no ledger exists on disk"
    assert ledger["exists"] is ledger_present_on_disk, (
        f"the record claims ledger exists={ledger['exists']} but the scan found "
        f"{observed['ledger_entries'] or 'no ledger entries'}"
    )
    assert ledger["exists"] is False
    assert ledger["entry_count"] == len(observed["ledger_entries"]) == 0
    assert ledger["head_sha256"] is None
    assert ledger["work_directory_created"] is False
    assert ledger["entry_path_convention"] == "<work-dir>/ledger/NNNNNNNN.json"
    assert ledger["marker_path_convention"] == "<work-dir>/markers/"


def test_no_accrual_or_seal_artifact_contradicts_the_unstarted_claim(
    record: dict[str, Any],
) -> None:
    """If real accrual lands, this record is stale and must be rewritten."""

    observed = scan_for_campaign_artifacts()
    execution = record["campaign_execution"]
    assert execution["state"] == "rendered-not-armed-not-started"
    assert not observed["namespace_accrual"], (
        "the record claims the campaign executed zero slots, but these in-repository "
        "accrual/seal artifacts exist: " + ", ".join(observed["namespace_accrual"])
    )
    assert not observed["ledger_entries"]


def test_action_counts_are_exact_integer_zeros(record: dict[str, Any]) -> None:
    execution = record["campaign_execution"]
    assert execution["action_counts_are_not_measurements"] is True
    for key in (
        "slots_executed",
        "sources_opened",
        "authorities_consumed",
        "probe_receipts_produced",
        "ledger_entries_written",
        "floor_evaluations_performed",
        "snapshots_captured",
        "packet_publication_attempts",
        "seal_process_launches",
        "semantic_reads",
    ):
        value = execution[key]
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"campaign_execution.{key} must be an integer action count"
        )
        assert value == 0, f"campaign_execution.{key} is {value}, but the campaign never ran"


def test_declared_authorities_are_unconsumed_and_match_the_frozen_plan(
    record: dict[str, Any], plan: dict[str, Any]
) -> None:
    execution = record["campaign_execution"]
    reserved = [reader["id"] for reader in plan["authority"]["reserved_readers_exactly"]]
    assert execution["reserved_reader_authorities_unconsumed"] == reserved
    assert execution["seal_authority_unconsumed"] == plan["authority"]["seal"]["id"]
    assert execution["plan_declared_status"] == plan["status"]


# ---------------------------------------------------------------------------
# The schedule, re-derived rather than trusted
# ---------------------------------------------------------------------------


def test_schedule_matches_the_frozen_plan_and_re_derives(
    record: dict[str, Any], plan: dict[str, Any]
) -> None:
    from datetime import datetime, timedelta, timezone

    schedule = record["schedule"]
    frozen = plan["schedule"]

    for key in (
        "anchor_at",
        "first_slot_at",
        "cadence_seconds",
        "grace_seconds",
        "first_slot_index",
        "last_slot_index",
        "slot_count",
        "final_selection_slot_at",
        "absolute_horizon_expires_at",
        "slot_formula",
        "backfill_allowed",
        "catch_up_allowed",
        "slot_replacement_allowed",
        "slot_skipping_allowed",
    ):
        assert schedule[key] == frozen[key], f"schedule.{key} drifted from the frozen plan"

    anchor = datetime(2026, 8, 17, tzinfo=timezone.utc)
    assert schedule["anchor_at"] == "2026-08-17T00:00:00Z"
    assert schedule["slot_count"] == 29
    assert schedule["cadence_seconds"] == 86400
    assert schedule["grace_seconds"] == 21600

    closes = anchor + timedelta(seconds=schedule["grace_seconds"])
    assert schedule["first_slot_closes_at"] == closes.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert schedule["first_slot_closes_at"] == "2026-08-17T06:00:00Z"

    final = anchor + timedelta(seconds=schedule["cadence_seconds"] * schedule["last_slot_index"])
    assert schedule["final_selection_slot_at"] == final.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert schedule["final_selection_slot_at"] == "2026-09-14T00:00:00Z"

    horizon = final + timedelta(seconds=schedule["grace_seconds"])
    assert schedule["absolute_horizon_expires_at"] == horizon.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert schedule["absolute_horizon_expires_at"] == "2026-09-14T06:00:00Z"

    assert schedule["last_slot_index"] - schedule["first_slot_index"] + 1 == schedule["slot_count"]


def test_record_was_generated_before_slot_zero_opened(record: dict[str, Any]) -> None:
    from datetime import datetime, timezone

    generated = datetime.strptime(record["generated_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    assert generated < datetime(2026, 8, 17, tzinfo=timezone.utc), (
        "the record claims an unstarted campaign but was generated after slot 0 opened"
    )


def test_missed_slot_consequence_is_terminal_with_no_recovery(
    record: dict[str, Any], plan: dict[str, Any]
) -> None:
    consequence = record["missed_slot_consequence"]
    frozen = plan["schedule"]

    assert consequence["applies_to_slot_0_exactly_as_to_slot_28"] is True
    assert "terminal" in consequence["outcome"]
    assert "schedule-integrity-failure" in consequence["outcome"]
    assert consequence["outcome"] in frozen["missed_slot_action"]
    for forbidden in (
        "catch_up",
        "backfill",
        "interpolation",
        "replacement_snapshot",
        "late_launch",
        "discretionary_skip",
        "later_probe_or_seal_authorized_after_a_miss",
    ):
        assert consequence[forbidden] is False, f"missed_slot_consequence.{forbidden} must be false"
    assert frozen["backfill_allowed"] is False
    assert frozen["catch_up_allowed"] is False
    assert "new namespace" in consequence["correction_path"]
    assert "2026-08-17T00:00:00Z" in consequence["statement"]
    assert consequence["driver_exit_status_for_terminal_failure"] == 4


# ---------------------------------------------------------------------------
# Machinery: enumerated by path and sha256, settled against disk
# ---------------------------------------------------------------------------


def test_every_enumerated_machinery_hash_matches_disk(record: dict[str, Any]) -> None:
    machinery = record["machinery"]
    entries = list(machinery["entries"]) + list(machinery["packet_manifest_recipe"])
    assert entries, "the machinery enumeration is empty"
    for entry in entries:
        path = ROOT / entry["path"]
        assert path.is_file(), f"{entry['path']} is enumerated but missing from disk"
        raw = path.read_bytes()
        assert entry["sha256"] == sha256_of(path), f"{entry['path']} sha256 drifted"
        assert entry["bytes"] == len(raw), f"{entry['path']} byte count drifted"
        assert entry["role"], f"{entry['path']} has no stated role"


def test_machinery_enumeration_covers_the_required_campaign_tools(
    record: dict[str, Any],
) -> None:
    enumerated = {entry["path"] for entry in record["machinery"]["entries"]}
    for required in (
        "scripts/ap_confirmatory_snapshot_v4.py",
        "scripts/ap_confirmatory_slot_v4.py",
        "scripts/v4_cadence.py",
    ):
        assert required in enumerated, f"{required} is not enumerated in the record"

    recipe = {entry["path"] for entry in record["machinery"]["packet_manifest_recipe"]}
    assert recipe == {
        "artifacts/animal-planet/evaluation/confirmatory-holdout-v4/recipe/build.py",
        "artifacts/animal-planet/evaluation/confirmatory-holdout-v4/recipe/verify.py",
    }


def test_machinery_is_actually_bound_in_the_packet_manifest(record: dict[str, Any]) -> None:
    """The binding claim is checked against the builder's own enumeration."""

    build_source = (NAMESPACE / "recipe" / "build.py").read_text(encoding="utf-8")
    machinery = record["machinery"]

    binding_key = machinery["packet_manifest_binding_key"]
    assert f'CAMPAIGN_TOOL_BINDING_KEY = "{binding_key}"' in build_source, (
        "the record names a packet-manifest binding key the builder does not define"
    )
    assert machinery["all_bound_in_packet_manifest"] is True
    assert set(machinery["packet_manifest_binds_by"]) == {"path", "sha256", "bytes"}

    for entry in machinery["entries"]:
        assert f'"{entry["path"]}"' in build_source, (
            f"{entry['path']} is claimed bound but is absent from CAMPAIGN_TOOL_PATHS"
        )


def test_this_records_own_test_stays_outside_the_campaign_tool_namespace(
    record: dict[str, Any],
) -> None:
    """This file must not masquerade as a campaign tool.

    ``build.py`` discovers campaign tools by *name shape* and fails closed with
    ``campaign_tool_enumeration_not_closed`` on any it finds unenumerated.  That
    net is aimed at tooling that touches evidence.  This suite touches none: it
    reads tracked artifacts, scans for absence, and runs the release validator.
    Naming it ``tests/test_v4_*.py`` would drag it into the campaign-tool set --
    and with it the builder enumeration, the verifier's ``BUILDER_SHA256``
    independence pin, and the packet suite's own oracle, none of which this node
    owns.  So the name is checked here, not left to chance.
    """

    own_path = str(Path(__file__).resolve().relative_to(ROOT))
    assert own_path == "tests/test_campaign_status_v4.py"
    assert record["machinery"]["this_records_own_test"] == own_path
    assert record["machinery"]["this_records_own_test_is_a_campaign_tool"] is False

    # Re-derive the builder's discovery rule rather than trusting the name.
    patterns = {
        "scripts": (("ap_confirmatory_", "_v4.py"), ("v4_", ".py")),
        "tests": (("test_ap_confirmatory_", "_v4.py"), ("test_v4_", ".py")),
    }

    def is_campaign_tool(directory: str, name: str) -> bool:
        for prefix, suffix in patterns[directory]:
            if name.startswith(prefix) and name.endswith(suffix):
                if len(name) > len(prefix) + len(suffix):
                    return True
        return False

    assert not is_campaign_tool("tests", Path(own_path).name), (
        f"{own_path} matches the campaign-tool name shape; the packet builder would "
        "fail closed unless a sibling-owned enumeration and hash pin were also changed"
    )
    # The rule is real: the name this file deliberately avoids would be caught.
    assert is_campaign_tool("tests", "test_v4_campaign_status.py")

    build_source = (NAMESPACE / "recipe" / "build.py").read_text(encoding="utf-8")
    assert f'"{own_path}"' not in build_source, (
        f"{own_path} is enumerated as a campaign tool but is not one"
    )


# ---------------------------------------------------------------------------
# The cadence unit is rendered and NOT activated
# ---------------------------------------------------------------------------


def test_cadence_unit_is_rendered_but_not_activated(record: dict[str, Any]) -> None:
    cadence = record["cadence_unit"]
    assert cadence["rendered"] is True
    assert cadence["activated"] is False
    assert cadence["installed"] is False
    assert cadence["armed"] is False
    assert "NOT activated" in cadence["statement"]

    unit_dir = ROOT / cadence["example_render_directory"]
    assert unit_dir.is_dir(), "the example render directory is missing"

    timers = sorted(unit_dir.glob("systemd/*.timer"))
    services = sorted(unit_dir.glob("systemd/*.service"))
    crontabs = sorted(unit_dir.glob("cron/*.crontab"))
    assert len(timers) == cadence["systemd_timer_units_rendered"] == 29
    assert len(services) == cadence["systemd_service_template_units_rendered"] == 1
    assert len(crontabs) == cadence["crontab_fallbacks_rendered"] == 1


def test_example_render_is_not_armable(record: dict[str, Any]) -> None:
    cadence = record["cadence_unit"]
    assert cadence["example_render_is_not_armable"] is True
    marker = cadence["example_render_paths_point_at"].rstrip("*")
    service = (ROOT / cadence["example_render_directory"] / "systemd").glob("*.service")
    for unit in service:
        text = unit.read_text(encoding="utf-8")
        exec_lines = [line for line in text.splitlines() if line.startswith("ExecStart=")]
        assert exec_lines, f"{unit.name} has no ExecStart"
        for line in exec_lines:
            assert marker in line, (
                f"{unit.name} ExecStart does not point at {marker}; the tracked example "
                "render must never reference a real path"
            )


def test_cadence_units_carry_the_no_backfill_guards(record: dict[str, Any]) -> None:
    cadence = record["cadence_unit"]
    assert cadence["refuse_manual_start"] is True
    assert cadence["persistent_replay_on_resume"] is False
    assert cadence["one_fresh_process_per_slot"] is True
    assert cadence["no_process_sleeps_or_polls_across_slots"] is True

    systemd = ROOT / cadence["example_render_directory"] / "systemd"
    service_text = "\n".join(
        unit.read_text(encoding="utf-8") for unit in sorted(systemd.glob("*.service"))
    )
    assert "RefuseManualStart=yes" in service_text
    for timer in sorted(systemd.glob("*.timer")):
        assert "Persistent=false" in timer.read_text(encoding="utf-8"), (
            f"{timer.name} would replay a missed firing, which is backfill"
        )


def test_first_timer_fires_at_slot_zero(record: dict[str, Any]) -> None:
    timer = ROOT / record["cadence_unit"]["example_render_directory"] / "systemd"
    text = (timer / "ap-confirmatory-v4-slot@0.timer").read_text(encoding="utf-8")
    assert "OnCalendar=2026-08-17 00:00:00 UTC" in text


# ---------------------------------------------------------------------------
# The arming procedure reproduces the runbook
# ---------------------------------------------------------------------------


def test_arming_procedure_is_bound_to_the_runbook_bytes(record: dict[str, Any]) -> None:
    procedure = record["operator_arming_procedure"]
    assert procedure["source"] == "docs/v4-campaign-runbook.md"
    assert procedure["source_sha256"] == sha256_of(RUNBOOK_PATH)
    assert procedure["source_bytes"] == len(RUNBOOK_PATH.read_bytes())
    assert procedure["arming_is_an_operator_decision_not_taken_here"] is True


def test_arming_steps_reproduce_the_runbook_commands(record: dict[str, Any]) -> None:
    runbook = RUNBOOK_PATH.read_text(encoding="utf-8")
    steps = record["operator_arming_procedure"]["steps"]
    assert [step["step"] for step in steps] == list(range(1, len(steps) + 1))

    # Every load-bearing fragment must actually appear in the runbook.
    for fragment in (
        'mkdir -p "$HOME/v4-campaign"',
        "python3 -B scripts/v4_cadence.py render",
        'python3 -B scripts/v4_cadence.py verify --unit-dir "$HOME/v4-cadence"',
        'python3 -B scripts/v4_cadence.py preflight --unit-dir "$HOME/v4-cadence"',
        'install -d "$HOME/.config/systemd/user"',
        'loginctl enable-linger "$USER"',
        "systemctl --user daemon-reload",
        "systemctl enable --now --user ap-confirmatory-v4-slot@{0..28}.timer",
        "systemctl --user list-timers --all 'ap-confirmatory-v4-slot@*'",
    ):
        assert fragment in runbook, f"{fragment!r} is not in the runbook"
        assert any(
            fragment in (step.get("command") or "") for step in steps
        ), f"{fragment!r} is in the runbook but missing from the recorded procedure"

    assert record["operator_arming_procedure"]["crontab_fallback_command"].strip() in runbook


def test_arming_record_carries_the_driver_exit_status_table(record: dict[str, Any]) -> None:
    runbook = RUNBOOK_PATH.read_text(encoding="utf-8")
    statuses = record["operator_arming_procedure"]["driver_exit_statuses"]
    assert set(statuses) == {"0", "1", "2", "3", "4", "5", "6"}
    assert "the campaign is over" in statuses["4"]
    for meaning in statuses.values():
        # The runbook writes the same table with a typographic dash.
        assert meaning.replace(" -- ", " — ") in runbook, f"{meaning!r} is not the runbook's wording"


def test_arming_record_states_the_terminal_cost_and_the_one_way_door(
    record: dict[str, Any], markdown: str
) -> None:
    procedure = record["operator_arming_procedure"]
    assert "29 consecutive days" in procedure["do_not_arm_unless"]
    assert "costs nothing" in procedure["disarming"]["before_slot_0_opens"]
    assert "terminally" in procedure["disarming"]["after_slot_0_opens"]
    assert "Persistent=false" in procedure["keep_the_machine_awake"]
    assert "one-way door" in markdown
    assert "no catch-up" in markdown


# ---------------------------------------------------------------------------
# P6 / P7 deferral
# ---------------------------------------------------------------------------


def test_deferral_matches_the_frozen_promotion_action(
    record: dict[str, Any], plan: dict[str, Any]
) -> None:
    deferral = record["deferral"]
    promotion = plan["promotion"]

    assert deferral["targets"] == ["P6", "P7"]
    assert deferral["status"] == "unconfirmed"
    assert deferral["promotion"] == "none"
    assert deferral["operational_promotion"] is False
    assert deferral["default_off_release_retained"] is True
    assert deferral["escalated"] is True

    action_key = "negative_invalid_inconclusive_timeout_or_missing_gate_action"
    assert deferral["plan_action_pointer"] == f"promotion.{action_key}"
    assert deferral["plan_action"] == promotion[action_key]
    assert deferral["plan_action"] == (
        "retain default-off release and escalate; no operational promotion"
    )

    context = deferral["plan_promotion_context"]
    for key, value in context.items():
        assert promotion[key] == value, f"promotion.{key} drifted from the frozen plan"


def test_automatic_versus_agent_repair_is_recorded_as_unimplemented(
    record: dict[str, Any],
) -> None:
    repair = record["deferral"]["automatic_versus_agent_repair"]
    assert repair["implemented"] is False
    assert repair["confirmed"] is False
    assert "unimplemented and unconfirmed" in repair["statement"]


def test_recorded_flags_are_actually_default_off(record: dict[str, Any]) -> None:
    """Read the claim, then prove it against the real policy resolver."""

    flags = record["deferral"]["flags_remain_default_off"]
    assert set(flags) == {"LM_RECALL_REPEAT_GATING", "LM_RECALL_REPEAT_DROP_TRAILING_STUBS"}
    assert all(value == "default-off" for value in flags.values())

    from living_memory.storage import FingerprintGatePolicy

    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in flags
    }
    policy = FingerprintGatePolicy.from_env(environment)
    assert policy.enabled is False, "LM_RECALL_REPEAT_GATING is not default-off"
    assert policy.drop_trailing_stubs is False, (
        "LM_RECALL_REPEAT_DROP_TRAILING_STUBS is not default-off"
    )


def test_prior_authorization_is_recorded_as_consumed(
    record: dict[str, Any], final_report: dict[str, Any]
) -> None:
    prior = record["deferral"]["prior_authorization"]
    assert prior["replacement_p6_evaluation_authorization_consumed"] is True
    assert prior["retry_permitted"] is False
    assert final_report["evidence"]["authorization_consumed"] is True
    assert final_report["evidence"]["retry_permitted"] is False


def test_deferral_is_scoped_to_the_automatic_lane(record: dict[str, Any]) -> None:
    deferral = record["deferral"]
    assert deferral["scope_of_deferral"] == "the automatic recall lane only"
    assert "automatic lane" in deferral["why_this_is_a_deferral_and_not_a_failure_of_the_whole_goal"]
    assert "not the path SPEC item numbering" in deferral["target_numbering"]


# ---------------------------------------------------------------------------
# Confirmed outcomes, checked against the final report
# ---------------------------------------------------------------------------


def resolve(document: dict[str, Any], pointer: str) -> Any:
    node: Any = document
    for part in pointer.split("."):
        node = node[part]
    return node


def test_confirmed_outcomes_match_the_final_report(
    record: dict[str, Any], final_report: dict[str, Any]
) -> None:
    confirmed = record["confirmed_on_held_out_evidence"]
    assert confirmed["source"] == "artifacts/animal-planet/evaluation/final-report.json"

    for block_name, keys in (
        (
            "payload_original_holdout",
            ("median_ratio", "p90_ratio", "top_result_retention", "useful_feedback_retention"),
        ),
        (
            "cross_scope_original_holdout",
            ("admission_reduction_ratio", "same_scope_retention_ratio"),
        ),
        (
            "correction_dominance",
            (
                "eval_replay_violations",
                "original_holdout_replay_violations",
                "focused_tests_passed",
                "focused_tests_failed",
            ),
        ),
        ("era_safety", ("focused_tests_passed", "focused_tests_failed")),
        (
            "shadow_latency",
            (
                "latency_p50_degradation_ratio",
                "latency_p95_degradation_ratio",
                "measured_requests_per_version",
            ),
        ),
    ):
        block = confirmed[block_name]
        source = resolve(final_report, block["source_pointer"])
        for key in keys:
            assert block[key] == source[key], (
                f"confirmed_on_held_out_evidence.{block_name}.{key} does not match "
                f"{block['source_pointer']}.{key}"
            )


def test_confirmed_outcomes_cite_only_passing_gates(
    record: dict[str, Any], final_report: dict[str, Any]
) -> None:
    gates = {gate["id"]: gate["status"] for gate in final_report["gates"]}
    confirmed = record["confirmed_on_held_out_evidence"]
    for block_name in (
        "payload_original_holdout",
        "cross_scope_original_holdout",
        "correction_dominance",
        "era_safety",
        "shadow_latency",
    ):
        block = confirmed[block_name]
        gate_id = block["gate"]
        assert gates[gate_id] == "pass", f"{block_name} cites gate {gate_id}, which did not pass"
        assert block["gate_status"] == "pass"

    cited = {confirmed[name]["gate"] for name in confirmed if isinstance(confirmed[name], dict)}
    failing = {gate_id for gate_id, status in gates.items() if status != "pass"}
    assert not (cited & failing), "a failing gate is presented among the confirmed outcomes"


def test_confirmed_outcomes_carry_the_recorded_fidelity_limits(
    record: dict[str, Any], final_report: dict[str, Any]
) -> None:
    recorded = record["confirmed_on_held_out_evidence"]["fidelity_limits"]
    assert recorded, "the confirmed outcomes drop their fidelity limits"
    for limit in recorded:
        assert limit in final_report["fidelity_limits"], (
            f"fidelity limit {limit!r} is not one the final report states"
        )


def test_retention_claims_are_not_rounded_up(record: dict[str, Any]) -> None:
    payload = record["confirmed_on_held_out_evidence"]["payload_original_holdout"]
    assert payload["top_result_retention"] == 1.0
    assert payload["useful_feedback_retention"] == 1.0
    assert payload["median_ratio"] < 0.60
    assert payload["p90_ratio"] < 0.60


# ---------------------------------------------------------------------------
# The pinned gating-off control arm
# ---------------------------------------------------------------------------


def test_release_validator_still_exits_zero(record: dict[str, Any]) -> None:
    control = record["control_arm"]
    assert control["validator"] == "python3 -B scripts/ap_release.py validate"
    assert control["exit_status"] == 0

    completed = subprocess.run(
        [sys.executable, "-B", "scripts/ap_release.py", "validate"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == control["exit_status"], (
        f"the record claims exit {control['exit_status']} but the validator exited "
        f"{completed.returncode}\nstdout: {completed.stdout}\nstderr: {completed.stderr}"
    )


def test_control_arm_pins_all_twenty_one_living_memory_modules(record: dict[str, Any]) -> None:
    control = record["control_arm"]
    manifest = json.loads(
        (EVAL / "release-v1" / "release-manifest.json").read_text(encoding="utf-8")
    )
    pinned = {
        path for path in manifest["implementation"] if path.startswith("src/living_memory/")
    }
    on_disk = {
        f"src/living_memory/{path.name}" for path in (ROOT / "src" / "living_memory").glob("*.py")
    }
    assert pinned == on_disk, "the pinned module set and the modules on disk disagree"
    assert len(pinned) == control["pinned_living_memory_module_count"] == 21
    assert control["src_living_memory_modified_by_this_record"] is False


# ---------------------------------------------------------------------------
# The markdown companion states the same thing as the JSON
# ---------------------------------------------------------------------------


def test_markdown_agrees_with_the_json_on_the_load_bearing_claims(
    record: dict[str, Any], markdown: str
) -> None:
    for fragment in (
        "rendered, not armed, not started",
        "executed zero slots",
        "2026-08-17T00:00:00Z",
        "2026-08-17T06:00:00Z",
        "2026-09-14T00:00:00Z",
        "2026-09-14T06:00:00Z",
        "not observed",
        "no ledger on disk",
        "retain default-off release and escalate; no operational promotion",
        "LM_RECALL_REPEAT_GATING",
        "LM_RECALL_REPEAT_DROP_TRAILING_STUBS",
        "unimplemented and unconfirmed",
        "deliberately **not** activated",
    ):
        assert fragment in markdown, f"{fragment!r} is missing from the markdown record"

    for entry in record["machinery"]["entries"]:
        assert entry["sha256"] in markdown or entry["path"] in markdown, (
            f"{entry['path']} is enumerated in the JSON but absent from the markdown"
        )


def test_markdown_states_no_floor_result_or_readiness(markdown: str) -> None:
    assert "Whether any of these is met is unknown." in markdown
    assert "aggregate-only" in markdown
    assert "claims no floor result, no readiness" in markdown
    lowered = markdown.lower()
    for forbidden in ("floor met", "floor satisfied", "readiness: ready", "packet sealed"):
        assert forbidden not in lowered, f"the markdown asserts {forbidden!r}"
