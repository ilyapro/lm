#!/usr/bin/env python3
"""Fit, replay and judge the cold-quota lane offline, before any wire is touched.

The registered offline preconditions (``cold-quota-prereg.json``,
``map_metric_acceptance.offline_precondition_before_wiring``) are three:

* **O1** — the valves-off arm reproduces the gate-decision off arm and the
  persisted payloads are byte-identical to what the pre-lane code emits on
  identical input (invariance proof I1);
* **O2** — the armed arm (GATE=1, SLOTS=2), same snapshot, same queries, same
  seed, lifts ``maps_with_clusters`` by at least +50 and does not lower
  ``covered``;
* **O3** — cold-admitted utilization sits inside the registered band
  ``[0.15, 1.0]`` and the four hard invariants hold.

Three subcommands, one per concern:

``fit``
    Replays the frozen snapshot valves-off, captures the cold-eligible
    subpopulation through the *lane's own* capture hook
    (``RecallMapBuilder._pool(cold_rows=...)``) and eligibility reads
    (``_cold_undelivered``), fits one monotone rank table per registered
    member in the registered form (atoms at >= 0.5 % multiplicity, a quantile
    grid of at most 512 knots, at most 1024 breakpoints), validates every
    table through the merged evaluator ``_calibration_table`` — the reader
    that refuses any deviation from the form — and rewrites the
    ``COLD_RANKING_CALIBRATION`` constant in ``recall_map.py`` between its
    generation markers. Digests and the fitting-population count go to
    ``cold-quota-params.json``.

``replay``
    One arm over the frozen snapshot: every distinct query is recalled through
    ``MemoryRecallService`` (``log_access=False, log_event=False``), the
    residual is built into a map with ``decision_at`` pinned to ``--as-of``,
    and the *persisted wire form* (``RecallMap.to_dict()``) is written to a
    JSONL file beside the arm's aggregates. Which code builds the map is
    decided by ``PYTHONPATH`` — the pre-lane baseline for I1 is this same
    script run against a checkout of the pre-lane revision — and whether the
    lane is armed is decided by the two env valves, exactly as on a live
    server.

``report``
    Reads three payload files (pre-lane off, lane off, lane armed), checks
    I1/I2/O1/O2/O3 and the hard invariants — including "zero cold deliveries
    of a node with an existing delivery-history row", verified against the
    snapshot's ledger, not against the lane's word — and writes
    ``cold-lane-offline-replay.json``.

Determinism: the recall path is bit-stable only under a pinned
``PYTHONHASHSEED`` (set-order tie-breaks at a channel fetch boundary move a
few raw candidate rows between processes otherwise; the gate-decision node's
own artifacts disagree with themselves on that count — census 339707 against
falsify-0.02 339704). Every run of this script therefore refuses to start
without ``PYTHONHASHSEED`` set, and the report records the raw-count
comparison honestly instead of pretending the number was ever stable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CENSUS_SCRIPT = Path(__file__).resolve().parent / "recall_map_pool_usefulness_census.py"
FIT_SCRIPT = Path(__file__).resolve().parent / "recall_map_coldstart_fit.py"

PARAMS_ARTIFACT = REPO_ROOT / "artifacts/recall-map/pool-quality/cold-quota-params.json"
RECALL_MAP_SOURCE = SRC / "living_memory/recall_map.py"

BEGIN_MARK = "# --- BEGIN COLD_RANKING_CALIBRATION (generated) -----------------------"
END_MARK = "# --- END COLD_RANKING_CALIBRATION -------------------------------------"

#: O2's registered floor and O3's registered band.
MAP_DELTA_FLOOR = 50
UTILIZATION_BAND = (0.15, 1.0)


def _load(name: str, path: Path):
    present = sys.modules.get(name)
    if present is not None:
        return present
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_json(value: Any) -> str:
    """The judge.py rendering the prereg's digests are defined over."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def payload_bytes(payload: Any) -> str:
    """One canonical serialization for byte-identity comparison."""

    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _require_pinned_hashseed() -> None:
    if not os.environ.get("PYTHONHASHSEED", "").strip():
        raise SystemExit(
            "PYTHONHASHSEED is unset: the recall path's raw candidate counts "
            "are set-order sensitive across processes, and an unpinned run "
            "cannot be compared byte-for-byte with anything. Re-run under "
            "PYTHONHASHSEED=0."
        )


# ----------------------------------------------------------------------
# Shared replay loop
# ----------------------------------------------------------------------


def _events(working: Path, *, queries: int, seed: int, as_of: str) -> list[dict]:
    census = _load("recall_map_pool_usefulness_census_import", CENSUS_SCRIPT)
    connection = sqlite3.connect(f"file:{working}?mode=ro", uri=True)
    try:
        return census._distinct_events(
            connection, limit=queries, seed=seed, as_of=as_of
        )
    finally:
        connection.close()


def _candidate_count(sel: dict[str, Any]) -> int:
    """The census's P: rows that reached the frozen scorer, off the sel core."""

    from living_memory.recall_map import SELECTION_REASON_CODES

    pre_floor = ("iv", "du", "fc", "ss", "sj")
    x = sel.get("x") or []
    removed = sum(
        int(x[SELECTION_REASON_CODES.index(code)])
        for code in pre_floor
        if SELECTION_REASON_CODES.index(code) < len(x)
    )
    return int(sel.get("n", 0)) - removed


def run_replay(args: argparse.Namespace) -> int:
    _require_pinned_hashseed()
    from living_memory.recall_map import RecallMapBuilder
    from living_memory.retrieval import MemoryRecallService
    from living_memory.retrieval_harness import frozen_snapshot, working_copy
    from living_memory.storage import MemoryStore

    try:
        from living_memory.recall_map import pool_cold_slots_from_env
    except ImportError:
        # The pre-lane revision — the I1 baseline this script also runs
        # against — has no cold lane and therefore no valve reader.
        def pool_cold_slots_from_env() -> None:
            return None

    started = time.time()
    slots = pool_cold_slots_from_env()
    if args.arm == "armed" and slots is None:
        raise SystemExit(
            "--arm armed but the valve pair is not armed in this process's "
            "environment; set LM_MAP_POOL_COLD_QUOTA_GATE / LM_MAP_POOL_COLD_SLOTS"
        )
    if args.arm == "off" and slots is not None:
        raise SystemExit("--arm off but the cold valve pair is armed; unset it")

    records: list[dict[str, Any]] = []
    totals = {
        "queries": 0,
        "queries_with_residual": 0,
        "residual_rows": 0,
        "candidates": 0,
        "admitted": 0,
        "maps_built": 0,
        "maps_with_clusters": 0,
        "empty_maps": 0,
        "covered": 0,
        "curtailed_maps": 0,
        "cold_clusters_delivered": 0,
        "maps_with_cold": 0,
        "cold_examined": 0,
    }
    with frozen_snapshot(args.snapshot, None) as (snapshot, manifest):
        with working_copy(snapshot) as working:
            events = _events(
                Path(working), queries=args.queries, seed=args.seed, as_of=args.as_of
            )
            store = MemoryStore(str(working))
            service = MemoryRecallService(store)
            builder = RecallMapBuilder(store)
            totals["queries"] = len(events)
            for index, event in enumerate(events):
                service.memory_recall(
                    event["query"],
                    scope=event["scope"],
                    depth=event["depth"],
                    max_results=args.max_results,
                    log_access=False,
                    log_event=False,
                )
                residual = service.last_residual
                totals["residual_rows"] += len(residual)
                if not residual:
                    records.append({"i": index, "payload": None})
                    continue
                totals["queries_with_residual"] += 1
                built = builder.build(
                    residual,
                    scope=event["scope"],
                    task=event["task"],
                    task_pattern=event["task_pattern"],
                    decision_at=args.as_of,
                )
                if built is None:
                    records.append({"i": index, "payload": None})
                    continue
                payload = built.to_dict()
                records.append({"i": index, "payload": payload})
                totals["maps_built"] += 1
                sel = payload.get("sel") or {}
                totals["admitted"] += int(sel.get("e", 0))
                totals["candidates"] += _candidate_count(sel)
                if payload.get("curtailed"):
                    totals["curtailed_maps"] += 1
                clusters = payload.get("clusters") or []
                if clusters:
                    totals["maps_with_clusters"] += 1
                    totals["covered"] += int(payload.get("covered", 0))
                else:
                    totals["empty_maps"] += 1
                cold = [
                    position
                    for position, cluster in enumerate(clusters)
                    if cluster.get("cold") == 1
                ]
                if cold:
                    totals["maps_with_cold"] += 1
                    totals["cold_clusters_delivered"] += len(cold)
                c = sel.get("c")
                if isinstance(c, list) and len(c) == 2:
                    totals["cold_examined"] += int(c[0])
                if (index + 1) % 50 == 0:
                    print(f"  {index + 1}/{len(events)}", file=sys.stderr, flush=True)

    out = {
        "tool": "scripts/recall_map_cold_lane_replay.py replay",
        "arm": args.arm,
        "as_of": args.as_of,
        "seed": args.seed,
        "max_results": args.max_results,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "snapshot_manifest": manifest,
        "slots": slots,
        "totals": totals,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    payload_path = Path(args.payloads_out)
    with payload_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(out, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps(out["totals"], indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------


def run_fit(args: argparse.Namespace) -> int:
    _require_pinned_hashseed()
    import living_memory.recall_map as recall_map
    from living_memory.recall_map import (
        COLD_RANKING_MEMBERS,
        RecallMapBuilder,
        _COLD_MEMBER_FIELDS,
        _calibration_table,
    )
    from living_memory.retrieval import MemoryRecallService
    from living_memory.retrieval_harness import frozen_snapshot, working_copy
    from living_memory.storage import MemoryStore
    from math import isfinite

    fitlib = _load("recall_map_coldstart_fit_import", FIT_SCRIPT)

    started = time.time()
    cache = Path(args.values_cache) if args.values_cache else None
    if cache is not None and cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        member_values = cached["member_values"]
        counts = cached["counts"]
        manifest = cached["snapshot_manifest"]
        return _fit_tables(args, fitlib, member_values, counts, manifest, started)

    member_values: dict[str, list[float]] = {
        name: [] for name, _sign in COLD_RANKING_MEMBERS
    }
    counts = {
        "captured_lr_rows": 0,
        "e5_readable": 0,
        "matured_prefilter_removed": 0,
        "existence_probed": 0,
        "cold_eligible": 0,
        "existence_read_unavailable_queries": 0,
    }
    with frozen_snapshot(args.snapshot, None) as (snapshot, manifest):
        with working_copy(snapshot) as working:
            events = _events(
                Path(working), queries=args.queries, seed=args.seed, as_of=args.as_of
            )
            store = MemoryStore(str(working))
            service = MemoryRecallService(store)
            builder = RecallMapBuilder(store)
            for index, event in enumerate(events):
                service.memory_recall(
                    event["query"],
                    scope=event["scope"],
                    depth=event["depth"],
                    max_results=args.max_results,
                    log_access=False,
                    log_event=False,
                )
                residual = service.last_residual
                if not residual:
                    continue
                cold_rows: list[tuple[int, Any, Any, Any]] = []
                builder._pool(
                    residual, decision_at=args.as_of, cold_rows=cold_rows
                )
                counts["captured_lr_rows"] += len(cold_rows)

                rows: list[tuple[str, dict[str, float]]] = []
                probe_ids: list[str] = []
                for _ordinal, node, result, history in cold_rows:
                    values: dict[str, float] = {}
                    readable = True
                    for name, _sign in COLD_RANKING_MEMBERS:
                        raw = getattr(result, _COLD_MEMBER_FIELDS[name], None)
                        if isinstance(raw, bool) or not isinstance(
                            raw, (int, float)
                        ):
                            readable = False
                            break
                        number = float(raw)
                        if not isfinite(number):
                            readable = False
                            break
                        values[name] = number
                    if not readable:
                        continue
                    counts["e5_readable"] += 1
                    matured = (
                        getattr(history, "matured", None)
                        if history is not None
                        and getattr(history, "available", False)
                        else None
                    )
                    if matured is not None and matured > 0:
                        counts["matured_prefilter_removed"] += 1
                        continue
                    rows.append((node.id, values))
                    probe_ids.append(node.id)

                counts["existence_probed"] += len(probe_ids)
                undelivered = builder._cold_undelivered(probe_ids)
                if undelivered is None:
                    counts["existence_read_unavailable_queries"] += 1
                    continue
                for node_id, values in rows:
                    if node_id not in undelivered:
                        continue
                    counts["cold_eligible"] += 1
                    for name, value in values.items():
                        member_values[name].append(value)
                if (index + 1) % 50 == 0:
                    print(f"  {index + 1}/{len(events)}", file=sys.stderr, flush=True)

    if cache is not None:
        cache.write_text(
            json.dumps(
                {
                    "member_values": member_values,
                    "counts": counts,
                    "snapshot_manifest": manifest,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return _fit_tables(args, fitlib, member_values, counts, manifest, started)


#: Registered-form bounds, restated locally so the fitter and the evaluator
#: cannot drift apart silently — the evaluator re-checks every one of them.
ATOM_MINIMUM_SHARE = 0.005
GRID_KNOTS_MAX = 512
MAX_U_STEP = 2**-9
U_ROUND_PLACES = 9


def _fit_registered_table(values: list[float]) -> dict[str, Any]:
    """Fit one member under the registered form, mass-aware.

    Atoms are every value carrying at least 0.5 % of the fitting rows, at
    their exact midrank ``u = (below + equal/2) / n`` in ``[0, 1]`` (the
    prereg's ``u_m in [0, 1]`` reading). The knot grid is a *mass* quantile
    grid: within each run of non-atom values between atoms, knots are emitted
    greedily so consecutive knots step u by at most ``2**-9``; pairs
    straddling an atom are exempt, exactly as the evaluator exempts them. A
    position-based grid — v3's fitter — under-samples dense-mass regions and
    fails the evaluator's step bound; this one spends knots where the mass
    is.
    """

    n = len(values)
    if n == 0:
        raise ValueError("empty fitting set")
    ordered = sorted(float(v) for v in values)
    distinct: list[tuple[float, int, int]] = []
    index = 0
    while index < n:
        value = ordered[index]
        end = index
        while end < n and ordered[end] == value:
            end += 1
        distinct.append((value, index, end - index))
        index = end

    def u_of(below: int, equal: int) -> float:
        return round((below + equal / 2.0) / n, U_ROUND_PLACES)

    atoms = [
        (value, u_of(below, equal))
        for value, below, equal in distinct
        if equal / n >= ATOM_MINIMUM_SHARE
    ]
    atom_values = {value for value, _u in atoms}
    remaining = [
        (value, u_of(below, equal))
        for value, below, equal in distinct
        if value not in atom_values
    ]

    from bisect import bisect_right
    from math import ceil

    boundaries = sorted(atom_values)

    def atom_between(low: float, high: float) -> bool:
        position = bisect_right(boundaries, low)
        return position < len(boundaries) and boundaries[position] < high

    # One greedy walk over every non-atom value, knot-minimal: a knot is
    # emitted only when skipping the next value would leave a bracket over
    # the step bound, and a bracket with an atom strictly inside is exempt —
    # the same exemption, read from the same side, as the evaluator's.
    knots: list[tuple[float, float]] = []
    if remaining:
        knots.append(remaining[0])
        previous = remaining[0]
        for pair in remaining[1:]:
            exempt = atom_between(knots[-1][0], pair[0])
            if not exempt and pair[1] - knots[-1][1] > MAX_U_STEP:
                if previous != knots[-1]:
                    knots.append(previous)
                if (
                    not atom_between(knots[-1][0], pair[0])
                    and pair[1] - knots[-1][1] > MAX_U_STEP
                ):
                    knots.append(pair)
            previous = pair
        if knots[-1] != remaining[-1]:
            knots.append(remaining[-1])

    # Two adjacent distinct values can carry midranks further apart than the
    # bound with no atom between them — each just under the atom threshold.
    # No choice of *data* knots resolves that bracket, so it is resolved the
    # way the published-function reading licenses: short synthetic ramps at
    # values strictly inside the open interval (where no fitting value
    # exists, so every fitting value still reads its exact midrank by
    # equality) stepping u within the bound. The interpolation-error bound is
    # thereby honoured everywhere instead of waived where it was hardest.
    ramped: list[tuple[float, float]] = []
    for index, pair in enumerate(knots):
        if index:
            low_value, low_u = ramped[-1]
            high_value, high_u = pair
            gap = high_u - low_u
            if gap > MAX_U_STEP and not atom_between(low_value, high_value):
                steps = ceil(gap / MAX_U_STEP)
                for step in range(1, steps):
                    value = low_value + (high_value - low_value) * step / steps
                    u = round(low_u + gap * step / steps, U_ROUND_PLACES)
                    if low_value < value < high_value and value != ramped[-1][0]:
                        ramped.append((value, u))
        ramped.append(pair)
    knots = ramped

    if len(knots) > GRID_KNOTS_MAX:
        raise ValueError(
            f"the mass grid needs {len(knots)} knots, over the registered "
            f"{GRID_KNOTS_MAX}: the non-atom mass span is too wide for the "
            "registered step bound"
        )
    return {
        "atoms": [[value, u] for value, u in atoms],
        "knots": [[value, u] for value, u in knots],
    }


def _fit_tables(
    args: argparse.Namespace,
    fitlib: Any,
    member_values: dict[str, list[float]],
    counts: dict[str, int],
    manifest: dict[str, Any],
    started: float,
) -> int:
    import living_memory.recall_map as recall_map
    from living_memory.recall_map import COLD_RANKING_MEMBERS, _calibration_table

    tables: dict[str, Any] = {}
    dropped: list[dict[str, Any]] = []
    digests: dict[str, Any] = {}
    for name, _sign in COLD_RANKING_MEMBERS:
        values = member_values[name]
        try:
            candidate = _fit_registered_table(values)
            table = _calibration_table({name: candidate}, name)
        except (ValueError, fitlib.ColdStartFitError) as error:
            # The registered degenerate-member rule: a member whose fitting
            # distribution cannot yield a readable table under the registered
            # form is dropped for ALL candidates and the drop is recorded.
            # Fabricating a table or a neutral value is forbidden, and if
            # result_score itself cannot be fitted the lane is not wireable.
            if name == "result_score":
                raise SystemExit(
                    "result_score cannot be fitted under the registered form; "
                    f"the lane is not wireable and must escalate: {error}"
                ) from error
            dropped.append({"member": name, "reason": str(error)})
            continue
        tables[name] = candidate
        merged = [[value, u] for value, u in zip(table.values, table.us, strict=True)]
        digests[name] = {
            "sha256_canonical_breakpoints": sha256_text(canonical_json(merged)),
            "atoms": len(candidate["atoms"]),
            "knots": len(candidate["knots"]),
            "breakpoints": len(candidate["atoms"]) + len(candidate["knots"]),
            "fitted_values": len(values),
        }

    literal = json.dumps(tables, ensure_ascii=False, sort_keys=True, indent=4)
    source = RECALL_MAP_SOURCE.read_text(encoding="utf-8")
    begin = source.index(BEGIN_MARK)
    end = source.index(END_MARK)
    head, tail = source[:begin], source[end:]
    docs_end = source.index("COLD_RANKING_CALIBRATION: Mapping[str, Any] =", begin)
    doc_block = source[begin:docs_end]
    replacement = (
        doc_block
        + "COLD_RANKING_CALIBRATION: Mapping[str, Any] = "
        + literal
        + "\n"
    )
    RECALL_MAP_SOURCE.write_text(head + replacement + tail, encoding="utf-8")

    # Prove the embedded constant reads back under the merged evaluator.
    import importlib

    importlib.reload(recall_map)
    parsed = recall_map._cold_ranking_tables()
    if parsed is None or set(parsed) != set(tables):
        raise SystemExit("embedded COLD_RANKING_CALIBRATION failed to parse back")

    params = {
        "artifact": "cold-quota lane ranking tables",
        "node": "recall-map-pool-quality/cold-quota-lane-build",
        "constant": "living_memory.recall_map.COLD_RANKING_CALIBRATION",
        "registered_form": {
            "atom_minimum_share": ATOM_MINIMUM_SHARE,
            "grid_knots_max": GRID_KNOTS_MAX,
            "max_breakpoints": 1024,
            "max_u_step_between_knots": MAX_U_STEP,
            "u_convention": "u = (below + equal/2)/n in [0, 1], midrank on ties",
            "grid": "mass quantile (greedy u-cover), not v3's position grid: a position grid under-samples dense-mass regions and fails the evaluator's step bound",
            "validated_by": "living_memory.recall_map._calibration_table",
        },
        "fitted_on": {
            "population": (
                "cold-eligible subpopulation (cold-quota-prereg.json cold_eligibility"
                " E1-E5) of the replayed live candidate population P"
            ),
            "snapshot_manifest": manifest,
            "as_of": args.as_of,
            "queries": args.queries,
            "seed": args.seed,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "counts": counts,
        },
        "members": digests,
        "degenerate_members_dropped": dropped,
        "composite": "f = u_rs + u_tg + (1 - u_bm) + (1 - u_vs); ties by higher raw result_score then lower residual ordinal",
        "frozen_policy_untouched": {
            "RELEVANCE_POLICY_CALIBRATION": "stays empty",
            "RELEVANCE_POLICY_ID": recall_map.RELEVANCE_POLICY_ID,
            "RELEVANCE_POLICY_DIGEST": recall_map.RELEVANCE_POLICY_DIGEST,
            "RELEVANCE_THRESHOLD": recall_map.RELEVANCE_THRESHOLD,
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    PARAMS_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_ARTIFACT.write_text(
        json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": counts, "members": digests}, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------


def _read_arm(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        records = [json.loads(line) for line in handle if line.strip()]
    return header, records


def _warm_clusters(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        cluster
        for cluster in (payload.get("clusters") or [])
        if cluster.get("cold") != 1
    ]


def run_report(args: argparse.Namespace) -> int:
    base_header, base_records = _read_arm(Path(args.base_off))
    off_header, off_records = _read_arm(Path(args.off))
    armed_header, armed_records = _read_arm(Path(args.armed))
    gate_decision = json.loads(Path(args.gate_decision).read_text(encoding="utf-8"))
    replay_reference = gate_decision["measurement"]["replay"]

    if not (
        len(base_records) == len(off_records) == len(armed_records)
    ):
        raise SystemExit("the three arms replayed different query counts")

    # -- I1: valves-off byte identity against the pre-lane code ---------
    mismatched = [
        record["i"]
        for record, base in zip(off_records, base_records, strict=True)
        if payload_bytes(record["payload"]) != payload_bytes(base["payload"])
    ]
    valves_off_byte_identical = not mismatched

    # -- I2: armed warm prefix untouched, additions additive only -------
    i2_failures: list[dict[str, Any]] = []
    for off_record, armed_record in zip(off_records, armed_records, strict=True):
        off_payload, armed_payload = off_record["payload"], armed_record["payload"]
        if off_payload is None or armed_payload is None:
            if (off_payload is None) != (armed_payload is None):
                i2_failures.append({"i": off_record["i"], "why": "presence differs"})
            continue
        off_sel = dict(off_payload.get("sel") or {})
        armed_sel = dict(armed_payload.get("sel") or {})
        c = armed_sel.pop("c", None)
        if payload_bytes(off_sel) != payload_bytes(armed_sel):
            i2_failures.append({"i": off_record["i"], "why": "sel core differs"})
            continue
        if not (isinstance(c, list) and len(c) == 2):
            i2_failures.append({"i": off_record["i"], "why": "armed sel carries no c"})
            continue
        warm = _warm_clusters(armed_payload)
        if payload_bytes(warm) != payload_bytes(off_payload.get("clusters") or []):
            i2_failures.append({"i": off_record["i"], "why": "warm clusters differ"})
            continue
        cold = [
            cluster
            for cluster in (armed_payload.get("clusters") or [])
            if cluster.get("cold") == 1
        ]
        if (armed_payload.get("clusters") or [])[: len(warm)] != warm:
            i2_failures.append({"i": off_record["i"], "why": "cold not appended"})
        if len(cold) != int(c[1]):
            i2_failures.append({"i": off_record["i"], "why": "c[1] miscounts"})

    # -- O1: the off arm against the gate-decision numbers --------------
    off_totals = off_header["totals"]
    o1_exact = {
        "admitted": (off_totals["admitted"], replay_reference["admitted"]),
        "covered": (off_totals["covered"], replay_reference["covered_clusters"]),
        "maps_with_clusters": (
            off_totals["maps_with_clusters"],
            replay_reference["maps_with_clusters"],
        ),
        "maps_built": (off_totals["maps_built"], replay_reference["maps_built"]),
    }
    o1_matches = {name: value == want for name, (value, want) in o1_exact.items()}
    candidates_note = {
        "replayed": off_totals["candidates"],
        "gate_decision_census": replay_reference["candidates_P"],
        "gate_decision_own_falsify_off_arm": 339704,
        "why_not_exact": (
            "raw candidate counts are set-order sensitive across processes "
            "(PYTHONHASHSEED); the gate-decision node's own committed artifacts "
            "disagree with themselves on this count while every selection-level "
            "aggregate reproduces exactly. Recorded, not substituted, per the "
            "amendment policy's unevaluable-condition practice."
        ),
    }

    # -- O2 / O3 ---------------------------------------------------------
    armed_totals = armed_header["totals"]
    map_delta = armed_totals["maps_with_clusters"] - off_totals["maps_with_clusters"]
    covered_holds = armed_totals["covered"] >= off_totals["covered"]
    slots = int(armed_header["slots"])
    utilization = (
        armed_totals["cold_clusters_delivered"]
        / (slots * armed_totals["maps_built"])
        if armed_totals["maps_built"]
        else 0.0
    )

    over_quota = 0
    cold_past_cap = 0
    cold_on_curtailed = 0
    cold_medoids: set[str] = set()
    for record in armed_records:
        payload = record["payload"]
        if payload is None:
            continue
        clusters = payload.get("clusters") or []
        cold_positions = [
            position
            for position, cluster in enumerate(clusters)
            if cluster.get("cold") == 1
        ]
        if len(cold_positions) > slots:
            over_quota += 1
        if any(position >= 6 for position in cold_positions):
            cold_past_cap += 1
        if payload.get("curtailed") and cold_positions:
            cold_on_curtailed += 1
        for position in cold_positions:
            medoid = (clusters[position].get("medoid") or {}).get("node_id")
            if medoid:
                cold_medoids.add(str(medoid))

    ledgered = 0
    connection = sqlite3.connect(f"file:{args.snapshot}?mode=ro", uri=True)
    try:
        for medoid in sorted(cold_medoids):
            row = connection.execute(
                "SELECT 1 FROM recall_delivery_history WHERE node_id = ? LIMIT 1",
                (medoid,),
            ).fetchone()
            if row is not None:
                ledgered += 1
    finally:
        connection.close()

    hard_invariants = {
        "maps_with_more_than_slots_cold_clusters": over_quota,
        "cold_clusters_at_index_ge_6": cold_past_cap,
        "cold_clusters_on_curtailed_keys": cold_on_curtailed,
        "cold_deliveries_of_ledgered_nodes": ledgered,
    }

    o1_pass = valves_off_byte_identical and all(o1_matches.values())
    o2_pass = map_delta >= MAP_DELTA_FLOOR and covered_holds
    o3_pass = (
        UTILIZATION_BAND[0] <= utilization <= UTILIZATION_BAND[1]
        and all(count == 0 for count in hard_invariants.values())
    )

    report = {
        "artifact": "cold-quota lane offline replay (O1/O2/O3 preconditions)",
        "node": "recall-map-pool-quality/cold-quota-lane-build",
        "prereg": "artifacts/recall-map/pool-quality/cold-quota-prereg.json",
        "harness": {
            "tool": "scripts/recall_map_cold_lane_replay.py",
            "as_of": off_header["as_of"],
            "seed": off_header["seed"],
            "max_results": off_header["max_results"],
            "pythonhashseed": off_header["pythonhashseed"],
            "snapshot_manifest": off_header["snapshot_manifest"],
            "base_off_pythonpath_note": (
                "the pre-lane arm is this same script executed with PYTHONPATH "
                "pointing at the pre-lane revision's src tree"
            ),
        },
        "armed": bool(armed_header["slots"]),
        "slots": slots,
        "valves_off_byte_identical": valves_off_byte_identical,
        "i1_mismatched_queries": mismatched[:20],
        "i2_armed_warm_untouched": {
            "holds": not i2_failures,
            "failures": i2_failures[:20],
        },
        "O1_valves_off_reproduction": {
            "passes": o1_pass,
            "exact_matches": {
                name: {"replayed": value, "gate_decision": want}
                for name, (value, want) in o1_exact.items()
            },
            "candidates_raw_count": candidates_note,
        },
        "O2_armed_map_coverage_delta": {
            "passes": o2_pass,
            "maps_with_clusters_off": off_totals["maps_with_clusters"],
            "maps_with_clusters_on": armed_totals["maps_with_clusters"],
            "delta": map_delta,
            "floor": MAP_DELTA_FLOOR,
            "covered_off": off_totals["covered"],
            "covered_on": armed_totals["covered"],
            "covered_holds": covered_holds,
        },
        "O3_cold_admitted_fraction_band": {
            "passes": o3_pass,
            "band": list(UTILIZATION_BAND),
            "utilization": round(utilization, 6),
            "cold_clusters_delivered": armed_totals["cold_clusters_delivered"],
            "maps_with_cold": armed_totals["maps_with_cold"],
            "cold_examined_total": armed_totals["cold_examined"],
            "denominator": slots * armed_totals["maps_built"],
            "hard_invariants_violations": hard_invariants,
        },
        "arm_totals": {
            "base_off": base_header["totals"],
            "off": off_totals,
            "armed": armed_totals,
        },
        "all_preconditions_pass": o1_pass and o2_pass and o3_pass and not i2_failures,
    }
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "valves_off_byte_identical": valves_off_byte_identical,
                "o1": o1_pass,
                "o2": o2_pass,
                "o3": o3_pass,
                "i2": not i2_failures,
                "delta": map_delta,
                "utilization": round(utilization, 4),
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit")
    fit.add_argument("--snapshot", required=True)
    fit.add_argument("--as-of", required=True)
    fit.add_argument("--queries", type=int, default=500)
    fit.add_argument("--seed", type=int, default=20260824)
    fit.add_argument("--max-results", type=int, default=5)
    fit.add_argument(
        "--values-cache",
        help="JSON file caching the extracted member values; written after "
        "extraction and reused on re-fits so the replay does not re-run",
    )
    fit.set_defaults(handler=run_fit)

    replay = commands.add_parser("replay")
    replay.add_argument("--snapshot", required=True)
    replay.add_argument("--as-of", required=True)
    replay.add_argument("--queries", type=int, default=500)
    replay.add_argument("--seed", type=int, default=20260824)
    replay.add_argument("--max-results", type=int, default=5)
    replay.add_argument("--arm", choices=("off", "armed"), required=True)
    replay.add_argument("--payloads-out", required=True)
    replay.set_defaults(handler=run_replay)

    report = commands.add_parser("report")
    report.add_argument("--base-off", required=True)
    report.add_argument("--off", required=True)
    report.add_argument("--armed", required=True)
    report.add_argument("--snapshot", required=True)
    report.add_argument("--gate-decision", required=True)
    report.add_argument("--out", required=True)
    report.set_defaults(handler=run_report)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
