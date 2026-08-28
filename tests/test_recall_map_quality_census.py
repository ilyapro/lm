"""The map-quality census: does it divide by the right thing, and stay quiet?

A census script has exactly two ways to be wrong, and they are independent.

* **The arithmetic.** ``sel`` was added to the payload after the column, so the
  map population and the sel population are different sets. Every ratio in the
  artifact must divide by the population it belongs to, and a payload with no
  ``sel`` block must not be folded in as a payload that inspected zero
  candidates. The corpus below is built so that *every* way of mixing the two
  denominators yields a different number from the right one — a test that only
  checked "the ratio is 0.022" would pass under several wrong implementations,
  so the assertions pin the counters as well as the quotient.
* **The mouth.** Cluster labels are forbidden output. The corpus carries
  labels that cannot occur by chance, and the test greps the serialized
  artifact for them rather than trusting the guard to have been called.

Everything is built through the production writers — ``recall_map._payload``,
``SelectionAccounting`` and ``MemoryStore.record_recall_event`` — so the census
reads the shape production actually persists. ``created_at`` is the one thing
rewritten afterwards, because the writer stamps it and the cutoff is under test.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from living_memory.postsession.usage_metric import PrivacyGuardError, check_privacy
from living_memory.recall_map import (
    SELECTION_REASON_CODES,
    MapCluster,
    MapMedoid,
    SelectionAccounting,
    _payload,
)
from living_memory.storage import MemoryStore

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "artifacts" / "recall-map" / "pool-quality" / "baseline.json"


def _load_census_module() -> Any:
    """Import the script by path; ``scripts/`` is not an importable package."""

    name = "recall_map_quality_census"
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered *before* execution: ``@dataclass`` resolves annotations
    # through ``sys.modules[cls.__module__]``, which is absent for a module
    # loaded by path alone.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


census = _load_census_module()

#: Labels that cannot occur by chance in an artifact of aggregates. If one of
#: these ever appears in the serialized report, the digest is not doing its job.
LABEL_A = "ZZQX-label-alpha-never-in-an-aggregate"
LABEL_B = "ZZQX-label-beta-never-in-an-aggregate"
LABEL_C = "ZZQX-label-gamma-never-in-an-aggregate"
HINT = "ZZQX-ask-hint-never-in-an-aggregate"

CUTOFF = "2026-03-01T00:00:00Z"


def _cluster(label: str) -> MapCluster:
    return MapCluster(
        label=label,
        count=3,
        medoid=MapMedoid(node_id="01ZZQXNODE0000000000000000", example="ZZQX-example"),
        ask_hint=HINT,
        plan_item=f"recall {label}",
        stage="stage3",
        member_ids=("01ZZQXNODE0000000000000000",),
    )


def _sel(inspected: int, admitted: int, excluded: tuple[int, ...]) -> SelectionAccounting:
    """Real accounting, so canonical trimming and the n == e + sum(x) law hold."""

    return SelectionAccounting(
        inspected=inspected, admitted=admitted, excluded=excluded
    )


#: (label, created_at, cluster labels, sel or None). One row per case the
#: denominators can be confused by.
SCENARIO: tuple[tuple[str, str, tuple[str, ...], SelectionAccounting | None], ...] = (
    # Before sel existed: in the map population, absent from the sel one.
    ("pre_sel_empty", "2026-02-01T00:00:00Z", (), None),
    ("pre_sel_full", "2026-02-01T01:00:00Z", (LABEL_A, LABEL_B), None),
    # A map with sel and no clusters: in the map and sel populations, out of
    # the label population.
    (
        "sel_empty",
        "2026-02-20T00:00:00Z",
        (),
        _sel(100, 0, (0, 0, 10, 0, 0, 90, 0)),
    ),
    # Same label set as pre_sel_full, so the digest must group all three.
    (
        "sel_full_frozen_width",
        "2026-02-28T00:00:00Z",
        (LABEL_A, LABEL_B),
        _sel(100, 2, (0, 0, 0, 0, 0, 98, 0)),
    ),
    # Reversed order: sorted() must make it the same set, and the vector is one
    # code wider because a usefulness gate fired.
    (
        "sel_full_gate_uf",
        "2026-02-28T12:00:00Z",
        (LABEL_B, LABEL_A),
        _sel(200, 8, (0, 0, 0, 0, 0, 180, 0, 12)),
    ),
    # A singleton label set at the full ledger width.
    (
        "sel_full_gate_nf",
        "2026-02-28T18:00:00Z",
        (LABEL_C,),
        _sel(100, 1, (0, 0, 0, 0, 0, 90, 0, 4, 5)),
    ),
    # Past the cutoff: must be invisible to every population.
    (
        "after_cutoff",
        "2026-03-02T00:00:00Z",
        (LABEL_C,),
        _sel(9000, 9000, (0, 0, 0, 0, 0, 0, 0)),
    ),
)


def _build_store(path: Path) -> None:
    """Write the scenario through the production writers, then pin the clock."""

    with MemoryStore(path) as store:
        stamped: list[tuple[str, str]] = []
        for name, created_at, labels, selection in SCENARIO:
            payload = _payload(
                [_cluster(label) for label in labels],
                pool_size=0 if not labels else 9,
                dropped=0,
                selection=selection,
            )
            event = store.record_recall_event(
                query=f"ZZQX query {name}",
                scope="global",
                recall_map=payload,
            )
            stamped.append((created_at, event.id))
    connection = sqlite3.connect(path)
    with connection:
        connection.executemany(
            "UPDATE recall_events SET created_at = ? WHERE id = ?", stamped
        )
    connection.close()


@pytest.fixture(scope="module")
def store_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("quality-census") / "corpus.sqlite3"
    _build_store(path)
    return path


@pytest.fixture(scope="module")
def report(store_path: Path) -> dict[str, Any]:
    return census.build_report(
        store_path,
        as_of=CUTOFF,
        salt=census.DEFAULT_LABEL_SALT,
        log=lambda _message: None,
    )


@pytest.fixture(scope="module")
def window(report: dict[str, Any]) -> dict[str, Any]:
    return report["windows"]["all"]


# ----------------------------------------------------------------------
# The arithmetic: three populations, never mixed
# ----------------------------------------------------------------------


def test_the_three_populations_are_different_sets_and_are_published_as_such(
    window: dict[str, Any]
) -> None:
    """If they were equal the rest of these tests could not discriminate."""

    populations = window["populations"]
    assert populations["maps"] == 6
    assert populations["sel_payloads"] == 4
    assert populations["non_empty_maps"] == 4
    # Pairwise distinct sets, not merely distinct counts: one sel payload is
    # empty and one non-empty map has no sel.
    assert populations["overlap_sel_and_empty_map"] == 1
    assert populations["overlap_sel_and_non_empty_map"] == 3


def test_admitted_fraction_divides_candidates_by_candidates(
    window: dict[str, Any]
) -> None:
    """``sum(sel.e) / sum(sel.n)``, over the sel population and nothing else."""

    admission = window["selector_admission"]
    assert admission["sel_payloads"] == 4
    assert admission["payloads_without_sel"] == 2
    assert admission["inspected"] == 500
    assert admission["admitted"] == 11
    assert admission["admitted_fraction"] == pytest.approx(11 / 500)
    assert admission["denominator"] == "inspected"


def test_a_payload_without_sel_is_unmeasured_and_never_a_zero(
    window: dict[str, Any], store_path: Path
) -> None:
    """The trap, stated as a test.

    Two of the six in-window payloads predate ``sel``. Folding them in as
    ``n == 0, e == 0`` leaves the quotient untouched but moves the payload
    count from 4 to 6, and dividing the admitted candidates by the *map* count
    instead of the inspected candidates gives 11/6 rather than 11/500. Both
    wrong readings are excluded here by number, not by intention.
    """

    admission = window["selector_admission"]
    assert admission["sel_payloads"] != window["populations"]["maps"]
    assert admission["admitted_fraction"] != pytest.approx(
        admission["admitted"] / window["populations"]["maps"]
    )
    # And the absent blocks really are absent, not zeroed on load.
    counters = census.LoadCounters()
    connection = census.open_readonly(store_path)
    try:
        records = census.load_records(
            connection, as_of=CUTOFF, salt="s", counters=counters
        )
    finally:
        connection.close()
    assert [record.sel_inspected for record in records].count(None) == 2
    assert 0 not in [record.sel_inspected for record in records]


def test_no_cluster_share_divides_by_the_map_population(
    window: dict[str, Any]
) -> None:
    presence = window["cluster_presence"]
    assert presence["maps"] == 6
    assert presence["maps_without_clusters"] == 2
    assert presence["share_without_clusters"] == pytest.approx(2 / 6)
    assert presence["denominator"] == "maps"
    # Not the sel population, and not the non-empty one.
    assert presence["share_without_clusters"] != pytest.approx(2 / 4)


def test_label_repeats_divide_by_the_non_empty_maps_and_group_by_sorted_set(
    window: dict[str, Any]
) -> None:
    """Three maps carry the same two labels, one of them in reverse order."""

    repeats = window["label_set_repeats"]
    assert repeats["non_empty_maps"] == 4
    assert repeats["distinct_label_sets"] == 2
    assert repeats["most_common"][0]["count"] == 3
    assert repeats["top1_share"] == pytest.approx(3 / 4)
    assert repeats["singleton_label_sets"] == 1
    assert repeats["maps_in_a_repeated_label_set"] == 3
    assert repeats["repeat_share"] == pytest.approx(3 / 4)
    assert repeats["denominator"] == "non_empty_maps"


def test_width_census_separates_the_frozen_vector_from_a_charged_gate(
    window: dict[str, Any]
) -> None:
    width = window["selection_vector_width"]
    assert width["by_width"] == {"7": 2, "8": 1, "9": 1}
    assert width["frozen_width"] == len(SELECTION_REASON_CODES) == 7
    assert width["payloads_at_frozen_width"] == 2
    assert width["payloads_with_a_pool_gate_charged"] == 2
    assert width["share_at_frozen_width"] == pytest.approx(2 / 4)
    # The gate codes only carry counts where the vector is wide enough to hold
    # them, and that is reported rather than read as a measured zero.
    assert width["denominator"] == "sel_payloads"
    admission = window["selector_admission"]
    assert admission["payloads_carrying_reason_code"]["uf"] == 2
    assert admission["payloads_carrying_reason_code"]["nf"] == 1
    assert admission["excluded_by_reason_code"]["uf"] == 16
    assert admission["excluded_by_reason_code"]["nf"] == 5


def test_the_selection_accounting_equation_is_checked_not_assumed(
    window: dict[str, Any], tmp_path: Path
) -> None:
    """n == e + sum(x) holds here, and a store where it does not says so."""

    assert window["selector_admission"]["accounting_holds"] is True

    broken = tmp_path / "broken.sqlite3"
    with MemoryStore(broken) as store:
        store.record_recall_event(
            query="ZZQX broken",
            scope="global",
            # Hand-built: SelectionAccounting would refuse to construct this.
            recall_map={
                "clusters": [],
                "pool": 1,
                "covered": 0,
                "sel": {"v": "r1", "n": 100, "e": 1, "x": [0, 0, 0, 0, 0, 1, 0], "o": 0},
            },
        )
    counters = census.LoadCounters()
    connection = census.open_readonly(broken)
    try:
        census.load_records(
            connection, as_of="2099-01-01T00:00:00Z", salt="s", counters=counters
        )
    finally:
        connection.close()
    assert counters.sel_accounting_violations == 1


def test_the_cutoff_excludes_a_later_payload_from_every_population(
    store_path: Path
) -> None:
    """The seventh row would move all four metrics if the cutoff leaked."""

    later = census.build_report(
        store_path,
        as_of="2026-03-03T00:00:00Z",
        salt=census.DEFAULT_LABEL_SALT,
        log=lambda _message: None,
    )["windows"]["all"]
    assert later["populations"]["maps"] == 7
    assert later["selector_admission"]["inspected"] == 9500
    assert later["selector_admission"]["admitted_fraction"] != pytest.approx(11 / 500)
    assert later["label_set_repeats"]["non_empty_maps"] == 5


def test_as_of_is_mandatory() -> None:
    with pytest.raises(SystemExit) as raised:
        census.main([])
    assert "--as-of is mandatory" in str(raised.value)


# ----------------------------------------------------------------------
# The mouth: no label text, ever
# ----------------------------------------------------------------------


def test_no_cluster_label_reaches_the_serialized_artifact(
    report: dict[str, Any]
) -> None:
    serialized = json.dumps(report, ensure_ascii=False)
    for forbidden in (LABEL_A, LABEL_B, LABEL_C, HINT, "ZZQX"):
        assert forbidden not in serialized


def test_the_report_passes_the_privacy_guard_and_the_guard_would_have_caught_it(
    report: dict[str, Any]
) -> None:
    """Both halves: the artifact is clean, and the guard is not a no-op here."""

    check_privacy(report)
    with pytest.raises(PrivacyGuardError):
        check_privacy({"label": "x" * 201})


def test_the_digest_groups_by_label_set_and_moves_with_the_salt() -> None:
    one = census.label_set_digest([LABEL_A, LABEL_B], salt="salt-one")
    reordered = census.label_set_digest([LABEL_B, LABEL_A], salt="salt-one")
    other_salt = census.label_set_digest([LABEL_A, LABEL_B], salt="salt-two")
    duplicated = census.label_set_digest([LABEL_A, LABEL_A], salt="salt-one")
    assert one == reordered
    assert one != other_salt
    # Duplicates are part of the set: two clusters sharing a label is a
    # different delivered map from one cluster carrying it.
    assert duplicated != census.label_set_digest([LABEL_A], salt="salt-one")
    assert len(one) == census.DIGEST_CHARS
    assert all(character in "0123456789abcdef" for character in one)


def test_the_salt_is_recorded_so_a_later_run_is_comparable(
    report: dict[str, Any]
) -> None:
    assert report["label_salt"] == census.DEFAULT_LABEL_SALT
    assert "after" in report["label_salt_note"]


# ----------------------------------------------------------------------
# The boundary: reading the store does not move a byte of it
# ----------------------------------------------------------------------


def test_the_census_does_not_write_to_the_store_it_reads(tmp_path: Path) -> None:
    path = tmp_path / "untouched.sqlite3"
    _build_store(path)
    # Any journal the writer left behind is gone before the fingerprint is
    # taken, so a WAL checkpoint cannot be mistaken for the census writing.
    before = path.read_bytes()
    census.build_report(
        path, as_of=CUTOFF, salt=census.DEFAULT_LABEL_SALT, log=lambda _m: None
    )
    assert path.read_bytes() == before


def test_the_census_never_instantiates_the_migrating_store() -> None:
    """``MemoryStore`` migrates whatever file it opens; the live one must not.

    Checked over the parse tree rather than the text, so the prose in the
    module docstring that *names* the prohibition does not satisfy it.
    """

    tree = ast.parse(
        (REPO / "scripts" / "recall_map_quality_census.py").read_text(encoding="utf-8")
    )
    imported: list[str] = []
    called: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name:
                called.append(name)
    assert "MemoryStore" not in imported
    assert "MemoryStore" not in called
    assert "open_readonly" in imported and "open_readonly" in called


# ----------------------------------------------------------------------
# The committed baseline
# ----------------------------------------------------------------------


def test_the_committed_baseline_carries_the_four_metrics_at_a_pinned_cutoff() -> None:
    assert BASELINE.is_file(), f"the baseline artifact is missing: {BASELINE}"
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    check_privacy(baseline)
    assert baseline["as_of"], "a published census must carry its cutoff"
    assert baseline["label_salt"]

    window = baseline["windows"]["all"]
    for metric, denominator in (
        ("cluster_presence", "maps"),
        ("selector_admission", "inspected"),
        ("label_set_repeats", "non_empty_maps"),
        ("selection_vector_width", "sel_payloads"),
    ):
        assert window[metric]["denominator"] == denominator

    # The trap, on the published numbers: the two denominators differ.
    assert (
        window["selector_admission"]["sel_payloads"]
        < window["cluster_presence"]["maps"]
    )
    assert window["selector_admission"]["payloads_without_sel"] > 0


def test_the_committed_baseline_reproduces_the_published_figures() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    check = baseline["cross_check"]
    assert check["reproduced"] is True, check["drifted"]
    assert check["compared"]["maps"]["cited"] == check["compared"]["maps"]["measured"]
    assert check["status"].startswith("informational")


def test_the_committed_baseline_shows_no_pool_gate_was_ever_charged() -> None:
    """The wire-level fact the whole parent goal turns on."""

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    width = baseline["windows"]["all"]["selection_vector_width"]
    assert width["payloads_with_a_pool_gate_charged"] == 0
    assert width["share_at_frozen_width"] == 1.0
