#!/usr/bin/env python3
"""Census of recall-map quality over the persisted ``recall_events.recall_map``.

Four numbers, three denominators
--------------------------------

The root goal names four live quantities. This script measures all four from
one column and nothing else — no ae journal, no inject stream, no dashboard
read — and it reports each against **its own** denominator:

1. ``cluster_presence`` — the share of delivered maps that carried no cluster
   at all, over the *map* population (every persisted payload in the window).
2. ``selector_admission`` — ``sum(sel.e) / sum(sel.n)``, the fraction of pool
   candidates the frozen relevance selector admitted, over the *sel* population
   (candidates, not payloads).
3. ``label_set_repeats`` — how concentrated the delivered label sets are, over
   the *non-empty map* population, counted through a salted digest so no label
   text is ever emitted.
4. ``selection_vector_width`` — the census of ``len(sel.x)``, over the *sel*
   population. Width equal to the frozen seven is the on-the-wire statement
   that neither env-gated pool valve was charged on the server that wrote the
   payload; a wider vector is the statement that one was.

THE TRAP THIS TOOL EXISTS TO NOT FALL INTO. ``sel`` was added to the payload
*after* the column, so only some payloads carry it: on the live store at the
pinned cutoff, 1262 of 2759. The map-count denominator and the sel denominator
are therefore different populations, and a payload without ``sel`` is *not* a
payload that inspected zero candidates. Every ratio here names the population
it divides by, ``populations`` publishes the overlaps between the three, and
nothing in this file ever divides a sel quantity by a map count. That is the
one arithmetic mistake this measurement can make, so it is checked rather than
assumed: :func:`measure_window` derives each ratio from the counter it belongs
to, and the tests fix a corpus where mixing them would give a different number.

``--as-of`` is mandatory, not optional: the live store grows continuously —
the map population itself grew by three payloads in the two minutes around the
cutoff this baseline is pinned at — so a run without a pinned cutoff is not
reproducible and must not be published.

READ-ONLY BY CONSTRUCTION. The database is opened only through
:func:`living_memory.replay.open_readonly` — a ``file:...?mode=ro`` URI — and
``MemoryStore`` is never instantiated anywhere in this script or its imports:
the store migrates whatever file it is pointed at, and pointing it at the live
database would rewrite the very thing this tool exists to describe. The one
symbol taken from :mod:`living_memory.recall_map` is the frozen reason-code
tuple, which is a literal.

AGGREGATES ONLY. Every value that reaches the artifact passes
:func:`living_memory.postsession.usage_metric.check_privacy` (printable ASCII,
at most 200 chars, no query text, no node content, no cluster label, no
ask_hint). Cluster labels are forbidden output under that guard, so the repeat
metric is emitted as *counts over a salted digest* of ``tuple(sorted(labels))``
with the salt recorded in the artifact. The salt is recorded on purpose and
buys exactly one thing: a later run — the "after" census the downstream sibling
writes — hashes the same label set to the same digest, so the two artifacts'
repeat concentrations are comparable. It is not a secret and is not claimed to
be one; what it buys the privacy guard is that the label *text* never leaves
the process. ``sel.q`` gists are never read at all.

Usage::

    PYTHONPATH=src python3 scripts/recall_map_quality_census.py \\
        --as-of 2026-08-24T12:43:56Z \\
        --out artifacts/recall-map/pool-quality/baseline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from living_memory.postsession.usage_metric import (  # noqa: E402
    DEFAULT_DB_PATH,
    PrivacyGuardError,
    check_privacy,
    normalize_instant,
    parse_timestamp,
)
from living_memory.recall_map import (  # noqa: E402
    SELECTION_LEDGER_REASON_CODES,
    SELECTION_REASON_CODES,
)
from living_memory.replay import open_readonly  # noqa: E402

#: Frozen ``sel.x`` width. A payload at exactly this width is a payload whose
#: server had neither pool valve charged: ``SelectionAccounting.__post_init__``
#: trims trailing zeros back to it, so "no gate fired" has exactly one
#: encoding and a wider vector cannot mean anything else.
FROZEN_WIDTH = len(SELECTION_REASON_CODES)

#: Widest the vector can legally get: the frozen seven plus the two gate codes.
LEDGER_WIDTH = len(SELECTION_LEDGER_REASON_CODES)

#: Default salt for the label-set digest. Recorded in every artifact this tool
#: writes, and stable across runs *by design* — see the module docstring: a
#: per-run salt would make the "before" and "after" repeat concentrations
#: incomparable, which is the only thing the metric is for.
DEFAULT_LABEL_SALT = "recall-map-pool-quality/live-map-quality-census/r1"

#: Hex characters kept from the digest. 16 hex characters is 64 bits; at the
#: order of a thousand distinct label sets the collision probability is ~1e-14,
#: and :func:`_label_repeats` checks for one anyway rather than assuming it.
DIGEST_CHARS = 16

#: Trailing sub-windows reported beside the full window, in hours. The map
#: population spans only the days since the feature was deployed, so an
#: all-window figure and a last-day figure are genuinely different readings:
#: on the baseline store they are 46.2% and 66.2% empty respectively, and a
#: reader who sees only the first would conclude the wrong trend.
TRAILING_HOURS: tuple[int, ...] = (24, 72)

#: What the parent goal published, and what this tool recomputes. Reproducing
#: these is a cross-check between two independent readings of one column: the
#: node that measured them and the tool that will be re-run after the fix.
#: Informational, never gating — no threshold anywhere reads it.
CITED_FIGURES: dict[str, Any] = {
    "maps": 2759,
    "maps_without_clusters": 1274,
    "share_without_clusters": 0.462,
    "sel_payloads": 1262,
    "inspected": 1302945,
    "admitted": 19725,
    "admitted_fraction": 0.0151,
    "sel_payloads_at_frozen_width": 1262,
    "non_empty_maps": 1485,
    "most_common_label_set_count": 237,
}

#: Where the cited figures were taken, and where they reproduce. The parent
#: names the instant it opened the store; the counts it published are the store
#: as it stood a little later, because three more map-bearing events landed
#: while the run was in flight. Both instants are recorded so the gap is a
#: stated fact rather than an unexplained mismatch.
CITED_OPENED_AT = "2026-08-24T12:42:00Z"
CITED_REPRODUCES_AT = "2026-08-24T12:43:56Z"


# ---------------------------------------------------------------------------
# One persisted payload, reduced to what the four metrics need
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MapRecord:
    """One ``recall_events.recall_map`` payload, with no text kept.

    Labels are consumed to produce ``label_digest`` inside :func:`load_records`
    and are never stored on this object, so no downstream step *can* emit them
    even by accident. ``sel.q`` gists are never read at all: they are the one
    part of the payload derived from node content.
    """

    created_at: str
    clusters: int
    curtailed: bool
    pool: int
    #: ``None`` for a map with no cluster: an empty label set is not a label set.
    label_digest: str | None
    #: ``None`` when the payload predates ``sel``. NOT zero — see the docstring.
    sel_inspected: int | None
    sel_admitted: int | None
    sel_excluded: tuple[int, ...] | None
    sel_version: str | None


@dataclass(slots=True)
class LoadCounters:
    """Payloads the map population could not admit, counted rather than hidden."""

    rows: int = 0
    unparseable: int = 0
    not_an_object: int = 0
    no_cluster_list: int = 0
    sel_malformed: int = 0
    sel_accounting_violations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_with_a_non_null_recall_map": self.rows,
            "excluded_unparseable_json": self.unparseable,
            "excluded_not_a_json_object": self.not_an_object,
            "excluded_no_cluster_list": self.no_cluster_list,
            "sel_objects_malformed": self.sel_malformed,
            "sel_accounting_equation_violations": self.sel_accounting_violations,
            "why": (
                "a payload this tool cannot read is a count, never a silent drop; "
                "the map population is rows minus the three excluded_* lines"
            ),
        }


def label_set_digest(labels: list[str], *, salt: str) -> str:
    """Salted digest of ``tuple(sorted(labels))``; the label text stays here.

    Duplicates are preserved — two clusters sharing a label is a different set
    from one cluster carrying it, and collapsing them would understate repeat
    concentration by exactly the maps the root goal is complaining about.
    """

    canonical = json.dumps(sorted(labels), ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(
        salt.encode("utf-8") + b"\x00" + canonical.encode("utf-8")
    ).hexdigest()
    return digest[:DIGEST_CHARS]


def _read_sel(
    payload: dict[str, Any], counters: LoadCounters
) -> tuple[int | None, int | None, tuple[int, ...] | None, str | None]:
    """Read the ``sel`` block, or report its absence as absence.

    Returning ``(None, None, None, None)`` for a payload without ``sel`` is the
    whole point: a payload that predates the block inspected an unknown number
    of candidates, not zero, and folding it in as zero would inflate the
    admitted fraction's denominator by nothing while inflating its payload
    count by one.
    """

    raw = payload.get("sel")
    if raw is None:
        return None, None, None, None
    if not isinstance(raw, dict):
        counters.sel_malformed += 1
        return None, None, None, None
    inspected = raw.get("n")
    admitted = raw.get("e")
    excluded = raw.get("x")
    if (
        isinstance(inspected, bool)
        or isinstance(admitted, bool)
        or not isinstance(inspected, int)
        or not isinstance(admitted, int)
        or not isinstance(excluded, list)
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in excluded)
    ):
        counters.sel_malformed += 1
        return None, None, None, None
    if inspected != admitted + sum(excluded):
        counters.sel_accounting_violations += 1
    version = raw.get("v")
    return (
        inspected,
        admitted,
        tuple(excluded),
        version if isinstance(version, str) else None,
    )


def load_records(
    connection: sqlite3.Connection, *, as_of: str, salt: str, counters: LoadCounters
) -> list[MapRecord]:
    """Every persisted map at or before ``as_of``, oldest first, read-only."""

    rows = connection.execute(
        "SELECT created_at, recall_map FROM recall_events "
        "WHERE recall_map IS NOT NULL AND created_at <= ? "
        "ORDER BY created_at, id",
        (as_of,),
    ).fetchall()

    records: list[MapRecord] = []
    for row in rows:
        counters.rows += 1
        try:
            payload = json.loads(row["recall_map"])
        except ValueError:
            counters.unparseable += 1
            continue
        if not isinstance(payload, dict):
            counters.not_an_object += 1
            continue
        clusters = payload.get("clusters")
        if not isinstance(clusters, list):
            counters.no_cluster_list += 1
            continue

        labels = [
            str(cluster.get("label") or "")
            for cluster in clusters
            if isinstance(cluster, dict)
        ]
        inspected, admitted, excluded, version = _read_sel(payload, counters)
        pool = payload.get("pool")
        records.append(
            MapRecord(
                created_at=str(row["created_at"]),
                clusters=len(clusters),
                curtailed=bool(payload.get("curtailed")),
                pool=pool if isinstance(pool, int) and not isinstance(pool, bool) else 0,
                label_digest=(
                    label_set_digest(labels, salt=salt) if clusters else None
                ),
                sel_inspected=inspected,
                sel_admitted=admitted,
                sel_excluded=excluded,
                sel_version=version,
            )
        )
    return records


# ---------------------------------------------------------------------------
# The four metrics. Each divides by its own population and says which.
# ---------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _cluster_presence(records: list[MapRecord]) -> dict[str, Any]:
    """Metric 1, over the map population."""

    maps = len(records)
    empty = [record for record in records if record.clusters == 0]
    curtailed = [record for record in records if record.curtailed]
    curtailed_empty = sum(1 for record in curtailed if record.clusters == 0)
    uncurtailed = maps - len(curtailed)
    uncurtailed_empty = len(empty) - curtailed_empty
    empty_with_empty_pool = sum(1 for record in empty if record.pool == 0)
    return {
        "denominator": "maps",
        "denominator_note": (
            "persisted recall_map payloads in the window; NOT the sel population"
        ),
        "maps": maps,
        "maps_without_clusters": len(empty),
        "share_without_clusters": _ratio(len(empty), maps),
        "curtailed": {
            "maps": len(curtailed),
            "without_clusters": curtailed_empty,
            "share_of_maps": _ratio(len(curtailed), maps),
            "why": (
                "a curtailed map carries no clusters by construction; it is a "
                "different defect from a map whose pool yielded nothing"
            ),
        },
        "uncurtailed": {
            "maps": uncurtailed,
            "without_clusters": uncurtailed_empty,
            "share_without_clusters": _ratio(uncurtailed_empty, uncurtailed),
        },
        "empty_with_empty_pool": empty_with_empty_pool,
        "empty_with_non_empty_pool": len(empty) - empty_with_empty_pool,
        "empty_pool_note": (
            "a map over an empty residual pool had nothing to describe; the rest "
            "had members and still delivered none"
        ),
    }


def _selector_admission(records: list[MapRecord]) -> dict[str, Any]:
    """Metric 2, over the sel population — candidates, not payloads."""

    carriers = [record for record in records if record.sel_inspected is not None]
    inspected = sum(record.sel_inspected or 0 for record in carriers)
    admitted = sum(record.sel_admitted or 0 for record in carriers)
    excluded_total = 0
    by_code: Counter[str] = Counter()
    carrying_code: Counter[str] = Counter()
    for record in carriers:
        vector = record.sel_excluded or ()
        excluded_total += sum(vector)
        for index, count in enumerate(vector):
            if index < LEDGER_WIDTH:
                by_code[SELECTION_LEDGER_REASON_CODES[index]] += count
                carrying_code[SELECTION_LEDGER_REASON_CODES[index]] += 1
    return {
        "denominator": "inspected",
        "denominator_note": (
            "sum(sel.n) over sel-carrying payloads only; NOT the map count and "
            "NOT the number of payloads"
        ),
        "sel_payloads": len(carriers),
        "payloads_without_sel": len(records) - len(carriers),
        "inspected": inspected,
        "admitted": admitted,
        "excluded": excluded_total,
        "admitted_fraction": _ratio(admitted, inspected),
        "accounting_holds": inspected == admitted + excluded_total,
        "excluded_by_reason_code": {
            code: by_code.get(code, 0) for code in SELECTION_LEDGER_REASON_CODES
        },
        "payloads_carrying_reason_code": {
            code: carrying_code.get(code, 0) for code in SELECTION_LEDGER_REASON_CODES
        },
        "reason_code_note": (
            "a code at zero excluded WITH zero payloads carrying it was never on "
            "the wire; a code at zero WITH payloads carrying it measured zero"
        ),
        # ``None`` becomes a name rather than a null key: check_privacy demands
        # string keys, and a payload whose ``sel`` carries no ``v`` is a real
        # state this census must be able to report instead of crashing on.
        "sel_versions": dict(
            sorted(
                Counter(record.sel_version or "unversioned" for record in carriers).items()
            )
        ),
    }


def _label_repeats(records: list[MapRecord]) -> dict[str, Any]:
    """Metric 3, over the non-empty map population, in digests only."""

    digests = [
        record.label_digest for record in records if record.label_digest is not None
    ]
    total = len(digests)
    counts = Counter(digests)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    repeated = sum(count for _, count in ordered if count > 1)
    singletons = sum(1 for _, count in ordered if count == 1)
    top = [
        {"digest": digest, "count": count, "share": _ratio(count, total)}
        for digest, count in ordered[:5]
    ]
    top5_count = sum(count for _, count in ordered[:5])
    return {
        "denominator": "non_empty_maps",
        "denominator_note": (
            "maps carrying at least one cluster; NOT the map count and NOT the "
            "sel population"
        ),
        "non_empty_maps": total,
        "distinct_label_sets": len(counts),
        "singleton_label_sets": singletons,
        "maps_in_a_repeated_label_set": repeated,
        "repeat_share": _ratio(repeated, total),
        "most_common": top,
        "top1_share": top[0]["share"] if top else None,
        "top5_share": _ratio(top5_count, total),
        "concentration_hhi": (
            round(sum((count / total) ** 2 for _, count in ordered), 6)
            if total
            else None
        ),
        "digest": {
            "over": "tuple(sorted(cluster labels)), duplicates preserved",
            "construction": "sha256(salt + 0x00 + canonical_json)[:16]",
            "why_digest": (
                "cluster labels are forbidden output under check_privacy; the "
                "digest is a grouping key, published with its salt so a later "
                "run hashes the same label set to the same value"
            ),
        },
    }


def _vector_width(records: list[MapRecord]) -> dict[str, Any]:
    """Metric 4, over the sel population."""

    carriers = [record for record in records if record.sel_excluded is not None]
    widths = Counter(len(record.sel_excluded or ()) for record in carriers)
    at_frozen = widths.get(FROZEN_WIDTH, 0)
    wider = sum(count for width, count in widths.items() if width > FROZEN_WIDTH)
    return {
        "denominator": "sel_payloads",
        "denominator_note": "sel-carrying payloads; NOT the map count",
        "sel_payloads": len(carriers),
        "by_width": {str(width): count for width, count in sorted(widths.items())},
        "frozen_width": FROZEN_WIDTH,
        "ledger_width": LEDGER_WIDTH,
        "payloads_at_frozen_width": at_frozen,
        "share_at_frozen_width": _ratio(at_frozen, len(carriers)),
        "payloads_with_a_pool_gate_charged": wider,
        "reason_codes": list(SELECTION_LEDGER_REASON_CODES),
        "why": (
            "the vector is canonically trimmed to the frozen width when no gate "
            "fired, so width == frozen_width is the payload's own statement that "
            "neither env valve was charged on the server that wrote it"
        ),
    }


def measure_window(
    records: list[MapRecord], *, label: str, start: str | None, end: str
) -> dict[str, Any]:
    """All four metrics over one window, each against its own denominator."""

    selected = [
        record
        for record in records
        if (start is None or record.created_at > start) and record.created_at <= end
    ]
    sel_carriers = sum(1 for record in selected if record.sel_inspected is not None)
    non_empty = sum(1 for record in selected if record.clusters > 0)
    empty_with_sel = sum(
        1
        for record in selected
        if record.clusters == 0 and record.sel_inspected is not None
    )
    return {
        "window": {
            "label": label,
            "start_exclusive": start,
            "end_inclusive": end,
            "first_payload": selected[0].created_at if selected else None,
            "last_payload": selected[-1].created_at if selected else None,
        },
        "populations": {
            "maps": len(selected),
            "sel_payloads": sel_carriers,
            "non_empty_maps": non_empty,
            "overlap_sel_and_empty_map": empty_with_sel,
            "overlap_sel_and_non_empty_map": sel_carriers - empty_with_sel,
            "why": (
                "three different denominators over one column: sel arrived after "
                "the column, so a map without sel is unmeasured, not zero. No "
                "ratio in this artifact mixes two of these populations."
            ),
        },
        "cluster_presence": _cluster_presence(selected),
        "selector_admission": _selector_admission(selected),
        "label_set_repeats": _label_repeats(selected),
        "selection_vector_width": _vector_width(selected),
    }


# ---------------------------------------------------------------------------
# The cross-check: two independent readings of one column
# ---------------------------------------------------------------------------


def _measured_figures(window: dict[str, Any]) -> dict[str, Any]:
    presence = window["cluster_presence"]
    admission = window["selector_admission"]
    repeats = window["label_set_repeats"]
    width = window["selection_vector_width"]
    top = repeats["most_common"]
    return {
        "maps": presence["maps"],
        "maps_without_clusters": presence["maps_without_clusters"],
        "share_without_clusters": presence["share_without_clusters"],
        "sel_payloads": admission["sel_payloads"],
        "inspected": admission["inspected"],
        "admitted": admission["admitted"],
        "admitted_fraction": admission["admitted_fraction"],
        "sel_payloads_at_frozen_width": width["payloads_at_frozen_width"],
        "non_empty_maps": repeats["non_empty_maps"],
        "most_common_label_set_count": top[0]["count"] if top else None,
    }


def cross_check(window: dict[str, Any], *, as_of: str) -> dict[str, Any]:
    """Compare the parent's published figures against this run's own reading.

    Rounded fields are compared at the precision the parent published them to;
    counts are compared exactly. Nothing here gates anything — a mismatch says
    the pinned cutoff is not the one those figures were taken at, which is a
    fact about the pin and not about the measurement.
    """

    measured = _measured_figures(window)
    compared: dict[str, Any] = {}
    drifted: list[str] = []
    for name, cited in CITED_FIGURES.items():
        value = measured.get(name)
        if isinstance(cited, float) and isinstance(value, int | float):
            digits = len(str(cited).partition(".")[2])
            agrees = round(float(value), digits) == cited
        else:
            agrees = value == cited
        compared[name] = {"cited": cited, "measured": value, "agrees": agrees}
        if not agrees:
            drifted.append(name)
    return {
        "why": (
            "the parent node and this tool read one column independently; equal "
            "counts mean they read it the same way"
        ),
        "cited_opened_at": CITED_OPENED_AT,
        "cited_reproduces_at": CITED_REPRODUCES_AT,
        "pin_note": (
            "the parent names the instant it opened the store; three more "
            "map-bearing events landed while its run was in flight, so its counts "
            "are the store as of the later instant. This run is pinned there."
        ),
        "as_of": as_of,
        "compared": compared,
        "reproduced": not drifted,
        "drifted": sorted(drifted),
        "status": "informational, never gating: no threshold in this tool reads it",
    }


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tilde(path: Path | str) -> str:
    text = str(path)
    home = str(Path.home())
    return f"~{text[len(home):]}" if text.startswith(home) else text


def _shift_hours(instant: str, hours: int) -> str:
    """``instant`` moved by ``hours``, in the exact shape the column stores."""

    parsed = parse_timestamp(instant)
    if parsed is None:
        raise ValueError(f"unusable instant: {instant!r}")
    shifted = parsed + timedelta(hours=hours)
    return shifted.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sel_epoch(records: list[MapRecord]) -> dict[str, Any]:
    """When ``sel`` started appearing, and how much of the column predates it."""

    carriers = [record for record in records if record.sel_inspected is not None]
    first = carriers[0].created_at if carriers else None
    before = (
        sum(1 for record in records if first is not None and record.created_at < first)
        if first
        else len(records)
    )
    return {
        "first_payload_carrying_sel": first,
        "payloads_before_that_instant": before,
        "payloads_at_or_after_without_sel": (
            len(records) - before - len(carriers) if first else 0
        ),
        "why": (
            "sel was added after the column: this is the mechanical reason the "
            "map denominator and the sel denominator are different populations"
        ),
    }


def build_report(
    db_path: Path,
    *,
    as_of: str,
    salt: str,
    log: Any,
) -> dict[str, Any]:
    """Load once, measure every window, cross-check, and guard before returning."""

    counters = LoadCounters()
    connection = open_readonly(db_path)
    try:
        records = load_records(connection, as_of=as_of, salt=salt, counters=counters)
        after = connection.execute(
            "SELECT count(*) FROM recall_events "
            "WHERE recall_map IS NOT NULL AND created_at > ?",
            (as_of,),
        ).fetchone()[0]
    finally:
        connection.close()
    log(f"loaded {len(records)} persisted maps at or before {as_of}")

    windows: dict[str, Any] = {
        "all": measure_window(records, label="all", start=None, end=as_of)
    }
    for hours in TRAILING_HOURS:
        name = f"trailing_{hours}h"
        windows[name] = measure_window(
            records, label=name, start=_shift_hours(as_of, -hours), end=as_of
        )

    report = {
        "artifact": "recall-map-pool-quality-baseline",
        "generated_at": _utc_now(),
        "as_of": as_of,
        "db": _tilde(db_path),
        "read_only": (
            "living_memory.replay.open_readonly (file:...?mode=ro); MemoryStore "
            "never instantiated"
        ),
        "privacy": (
            "aggregates only; every value passes usage_metric.check_privacy. No "
            "cluster label, ask_hint, query or node id reaches this artifact."
        ),
        "label_salt": salt,
        "label_salt_note": (
            "recorded on purpose: an 'after' run must hash the same label set to "
            "the same digest for the repeat concentrations to be comparable"
        ),
        "source": "recall_events.recall_map, and no other input",
        "cutoff": {
            "as_of": as_of,
            "as_of_is_mandatory": (
                "the live store grows continuously; an unpinned run is not "
                "reproducible and must not be published"
            ),
            "payloads_after_as_of": after,
        },
        "load": counters.to_dict(),
        "sel_epoch": _sel_epoch(records),
        "windows": windows,
        "cross_check": cross_check(windows["all"], as_of=as_of),
        "notes": _notes(windows, counters),
    }
    check_privacy(report)
    return report


def _notes(windows: dict[str, Any], counters: LoadCounters) -> list[str]:
    """What a reader has to know that the numbers do not say."""

    every = windows["all"]
    presence = every["cluster_presence"]
    admission = every["selector_admission"]
    width = every["selection_vector_width"]
    notes = [
        "Three denominators, never mixed: maps "
        f"({presence['maps']}), sel-carrying payloads "
        f"({admission['sel_payloads']}) and non-empty maps "
        f"({every['label_set_repeats']['non_empty_maps']}). A payload without a "
        "sel block inspected an unknown number of candidates, not zero.",
        "The persisted payload carries no cache key, so this census cannot split "
        "the no-cluster share by chat key versus node key; it reports the pooled "
        "share and the curtailed/uncurtailed split instead.",
    ]
    last_day = windows.get(f"trailing_{TRAILING_HOURS[0]}h")
    if last_day and last_day["cluster_presence"]["share_without_clusters"] is not None:
        notes.append(
            "The no-cluster share is not flat in time: "
            f"{presence['share_without_clusters']} over the whole window against "
            f"{last_day['cluster_presence']['share_without_clusters']} over the "
            f"trailing {TRAILING_HOURS[0]}h. Compare like windows, not one against "
            "the other."
        )
    if width["payloads_with_a_pool_gate_charged"] == 0 and width["sel_payloads"]:
        notes.append(
            f"Every one of the {width['sel_payloads']} sel-carrying payloads is at "
            f"the frozen width {FROZEN_WIDTH}: on the wire, neither pool valve was "
            "charged on any server that wrote into this store."
        )
    dominant = max(
        admission["excluded_by_reason_code"].items(), key=lambda item: item[1]
    )
    if dominant[1]:
        notes.append(
            f"Exclusions are not spread across the reason codes: {dominant[0]} "
            f"accounts for {dominant[1]} of {admission['excluded']}."
        )
    if counters.sel_accounting_violations:
        notes.append(
            f"{counters.sel_accounting_violations} sel objects fail their own "
            "accounting equation n == e + sum(x); they are counted, not repaired."
        )
    return notes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--as-of",
        help="ISO instant; cuts recall_events on created_at. MANDATORY: without "
        "it a run is not reproducible and must not be published",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH.expanduser()),
        help="database to read (opened read-only; never written)",
    )
    parser.add_argument(
        "--label-salt",
        default=DEFAULT_LABEL_SALT,
        help="salt for the label-set digest; recorded in the artifact "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--out", metavar="PATH", help="write the JSON report here (default: stdout)"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress progress on stderr"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.as_of or not str(args.as_of).strip():
        raise SystemExit(
            "--as-of is mandatory: the live store grows continuously, so a run "
            "without a pinned cutoff is not reproducible and must not be published"
        )
    try:
        as_of = normalize_instant(args.as_of)
    except ValueError as exc:
        raise SystemExit(f"--as-of: {exc}") from exc

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        raise SystemExit(f"database not found: {db_path}")

    def log(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr, flush=True)

    try:
        report = build_report(db_path, as_of=as_of, salt=args.label_salt, log=log)
    except PrivacyGuardError as exc:
        raise SystemExit(str(exc)) from exc
    except (ValueError, OSError, sqlite3.Error) as exc:
        raise SystemExit(f"census failed: {exc}") from exc

    # Second, independent pass: the guard runs where the report is built and
    # again here, before a single byte reaches disk.
    check_privacy(report)
    payload = json.dumps(report, indent=2, sort_keys=True)

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(payload)

    for name, window in report["windows"].items():
        presence = window["cluster_presence"]
        admission = window["selector_admission"]
        repeats = window["label_set_repeats"]
        width = window["selection_vector_width"]
        print(
            "%-16s maps=%s no-cluster=%s (%s) | sel payloads=%s admitted=%s/%s (%s) "
            "| non-empty=%s top1=%s repeat=%s | widths=%s"
            % (
                name,
                presence["maps"],
                presence["maps_without_clusters"],
                presence["share_without_clusters"],
                admission["sel_payloads"],
                admission["admitted"],
                admission["inspected"],
                admission["admitted_fraction"],
                repeats["non_empty_maps"],
                repeats["top1_share"],
                repeats["repeat_share"],
                width["by_width"],
            )
        )
    check = report["cross_check"]
    print(
        "cross-check against the published figures: %s%s"
        % (
            "reproduced" if check["reproduced"] else "DRIFTED",
            "" if check["reproduced"] else " (" + ", ".join(check["drifted"]) + ")",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
