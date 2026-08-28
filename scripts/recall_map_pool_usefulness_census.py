#!/usr/bin/env python3
"""The pool-candidate ``usefulness_score`` distribution, replayed not proxied.

What this measures, and why nothing cheaper would do
----------------------------------------------------

``LM_MAP_POOL_MIN_USEFULNESS`` is a floor applied inside
``RecallMapBuilder._pool``, at ``recall_map.py:1949``, to the rows that have
already survived the identity, duplicate and ballast checks and have not yet
had a single delivery-history row fetched. That set — call it **P** — is the
only population a floor can honestly be read off, and it is not any of the
distributions that are cheap to compute:

* the **corpus-wide** distribution is 70.8 % exactly ``0.0`` with ``p90 = 0.116``;
  a floor read off it deletes nearly every node there is.
* the **recall-returned** distribution (``p20 = 0.0777``) is the nodes recall
  *delivered* — the complement of the residual the map is built out of.

So P is obtained by replay: real ``recall_events`` queries are run against a
frozen snapshot of the live store through the same ``MemoryRecallService`` the
server uses, ``last_residual`` is taken, and ``_pool``'s own classification
pass is run over it.

The classification pass is *mirrored*, not called
------------------------------------------------

``_pool`` does not expose its intermediate candidate list, and this node may
not edit ``recall_map.py`` to make it. The loop below is therefore a mirror of
``recall_map.py:1928-1953`` that calls the module's own ``_ballast_reason``
rather than reimplementing it — the same choice ``recall_map_field_extract.py``
made for stage attribution.

A mirror that has drifted is worse than no mirror, so every replayed query
asserts the mirror against the real thing: ``_pool`` is *also* called, and its
``SelectionAccounting`` — which is unconditional and which
``__post_init__`` proves covers the whole residual — must satisfy

    len(candidates) == inspected - (iv + du + fc + ss + sj)

Any disagreement aborts the run rather than being reported. That check is what
makes "these are the rows the gate would actually see" a claim and not a hope.

Read-only, and the live database is never opened by this script
---------------------------------------------------------------

``MemoryStore`` migrates and writes whatever it opens, so the replay runs
against a temporary working copy of a frozen snapshot
(``retrieval_harness.working_copy``). Recalls are issued with
``log_access=False, log_event=False``: the working copy is not mutated, and
the builder's corpus-revision probe stays put.

The falsifier
-------------

``--falsify t`` replays the *same* query set against the *same* snapshot twice,
once with the valve unset and once with it charged at ``t``, and reports
``RecallMap.covered`` summed over the replay together with the count of
non-empty maps. The pre-registered band is one-sided and lives in
``artifacts/recall-map/follow-signal/procedure.md`` §7.6: a gate may not reduce
coverage.

Typical use::

    PYTHONPATH=src python3 scripts/recall_map_pool_usefulness_census.py \
        --source-db ~/.local/share/living-memory/global.sqlite3 \
        --as-of 2026-08-24T13:00:00Z --queries 300 \
        --out /tmp/pool-census.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from living_memory.recall_map import (  # noqa: E402
    SELECTION_REASON_CODES,
    RecallMapBuilder,
    _ballast_reason,
    pool_usefulness_floor_from_env,
)
from living_memory.retrieval import MemoryRecallService  # noqa: E402
from living_memory.retrieval_harness import frozen_snapshot, working_copy  # noqa: E402
from living_memory.storage import MemoryStore  # noqa: E402

#: Positions of the pre-floor exclusion codes inside the frozen vector. The
#: mirror below must account for exactly these and no others, because the floor
#: sits after them and before ``lr``.
_PRE_FLOOR_CODES = ("iv", "du", "fc", "ss", "sj")

#: The production recall cut (``server.py:1040``). The residual is everything
#: past it, so this argument decides where P begins.
DEFAULT_MAX_RESULTS = 5

BUDGET_SHARE = 0.20
"""§4.3 condition 2, inherited unchanged: a threshold removes at most 20%."""


# ----------------------------------------------------------------------
# Query selection — the latency bench's recipe, for the same reasons
# ----------------------------------------------------------------------


def _parse_depth(raw: Any) -> int | str:
    if raw is None:
        return 1
    text = str(raw)
    try:
        return int(text)
    except ValueError:
        return text


def _distinct_events(
    connection: sqlite3.Connection, *, limit: int, seed: int, as_of: str
) -> list[dict[str, Any]]:
    """Distinct real queries with the scope, depth and task they ran under.

    Distinct on the query text: replaying one string twice measures the page
    cache, not the pool. ``task`` and ``task_pattern`` are replayed because
    together with the scope they are the builder's cache key, and the cache key
    is what makes a curtail verdict — and therefore a map's emptiness — real.
    """

    columns = {row[1] for row in connection.execute("PRAGMA table_info(recall_events)")}
    pattern = "task_pattern" if "task_pattern" in columns else "NULL AS task_pattern"
    rows = connection.execute(
        f"""
        SELECT query,
               requested_scope,
               depth,
               task,
               {pattern},
               MAX(created_at) AS created_at
        FROM recall_events
        WHERE LENGTH(TRIM(query)) > 0 AND created_at <= ?
        GROUP BY query
        ORDER BY created_at
        """,
        (as_of,),
    ).fetchall()
    events = [
        {
            "query": row[0],
            "scope": row[1] or "global",
            "depth": _parse_depth(row[2]),
            "task": row[3],
            "task_pattern": row[4],
            "created_at": row[5],
        }
        for row in rows
    ]
    if len(events) <= limit:
        return events
    return random.Random(seed).sample(events, limit)


# ----------------------------------------------------------------------
# The mirrored classification pass
# ----------------------------------------------------------------------


def _candidates(results: Sequence[Any]) -> list[Any]:
    """Mirror of ``recall_map.py:1928-1953`` up to the floor, exclusive.

    Returns the nodes in residual order that reach the ``usefulness_floor``
    branch. Cross-checked against ``SelectionAccounting`` by every caller.
    """

    seen: set[str] = set()
    candidates: list[Any] = []
    for result in results:
        node = getattr(result, "node", None)
        node_id = getattr(node, "id", None) if node is not None else None
        if not isinstance(node_id, str) or not node_id.strip():
            continue
        if node_id in seen:
            continue
        seen.add(node_id)
        reason = _ballast_reason(
            getattr(node, "content", None),
            getattr(node, "context", None),
            getattr(node, "provenance", None),
        )
        if reason is not None:
            continue
        candidates.append(node)
    return candidates


def _pre_floor_excluded(accounting: Any) -> int:
    """How many rows the frozen vector says never reached the floor."""

    excluded = accounting.excluded
    total = 0
    for code in _PRE_FLOOR_CODES:
        index = SELECTION_REASON_CODES.index(code)
        if index < len(excluded):
            total += int(excluded[index])
    return total


def _usefulness(node: Any) -> float | None:
    """The score exactly as ``_below_usefulness`` would read it off the node."""

    value = getattr(node, "usefulness_score", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return None if score != score else score


# ----------------------------------------------------------------------
# One replay arm
# ----------------------------------------------------------------------


def replay(
    service: MemoryRecallService,
    store: MemoryStore,
    events: Sequence[dict[str, Any]],
    *,
    as_of: str,
    max_results: int,
    collect_pool: bool,
    log: Any,
) -> dict[str, Any]:
    """One pass over the query set. Returns the arm's aggregates.

    ``collect_pool`` also gathers per-candidate rows — score, cold flag,
    admitted flag — which is what the decile table is built from. The
    falsifier arm does not need them and does not pay for them.
    """

    builder = RecallMapBuilder(store)
    floor = pool_usefulness_floor_from_env()
    rows: list[dict[str, Any]] = []
    covered = 0
    maps_with_clusters = 0
    maps_built = 0
    empty_maps = 0
    residual_total = 0
    admitted_total = 0
    candidate_total = 0
    queries_with_residual = 0
    mirror_checks = 0

    for index, event in enumerate(events):
        service.memory_recall(
            event["query"],
            scope=event["scope"],
            depth=event["depth"],
            max_results=max_results,
            log_access=False,
            log_event=False,
        )
        residual = service.last_residual
        residual_total += len(residual)
        if not residual:
            continue
        queries_with_residual += 1

        pool, accounting = builder._pool(residual, decision_at=as_of)
        admitted_total += len(pool)

        # The mirror check. With the valve armed the floor removes rows the
        # mirror still lists, so the identity holds only against the arm that
        # is not gated; the gated arm checks the weaker containment instead.
        candidates = _candidates(residual)
        candidate_total += len(candidates)
        expected = accounting.inspected - _pre_floor_excluded(accounting)
        if floor is None:
            if len(candidates) != expected:
                raise SystemExit(
                    "mirror of _pool's classification pass disagrees with the "
                    f"frozen accounting on query {index}: mirror={len(candidates)} "
                    f"accounting={expected}"
                )
        elif len(candidates) < expected:
            raise SystemExit(
                "gated arm removed rows the mirror never saw on query "
                f"{index}: mirror={len(candidates)} accounting={expected}"
            )
        mirror_checks += 1

        if collect_pool:
            admitted_ids = {member.node.id for member in pool}
            histories = builder._matured_history(
                [node.id for node in candidates], decision_at=as_of
            )
            for node in candidates:
                history = histories.get(node.id)
                available = bool(getattr(history, "available", False))
                matured = getattr(history, "matured", None) if available else None
                rows.append(
                    {
                        "score": _usefulness(node),
                        "cold": available and matured == 0,
                        "history_unavailable": not available,
                        "admitted": node.id in admitted_ids,
                    }
                )

        built = builder.build(
            residual,
            scope=event["scope"],
            task=event["task"],
            task_pattern=event["task_pattern"],
            decision_at=as_of,
        )
        if built is None:
            continue
        maps_built += 1
        if built.clusters:
            maps_with_clusters += 1
            covered += int(getattr(built, "covered", 0) or 0)
        else:
            empty_maps += 1
        if (index + 1) % 25 == 0:
            log(f"  {index + 1}/{len(events)} queries replayed")

    return {
        "floor_in_effect": floor,
        "queries": len(events),
        "queries_with_residual": queries_with_residual,
        "mirror_checks_passed": mirror_checks,
        "residual_rows": residual_total,
        "candidates": candidate_total,
        "admitted": admitted_total,
        "maps_built": maps_built,
        "maps_with_clusters": maps_with_clusters,
        "empty_maps": empty_maps,
        "covered": covered,
        "rows": rows,
    }


# ----------------------------------------------------------------------
# The decile table and the rule
# ----------------------------------------------------------------------


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    """Nearest-rank quantile. No interpolation: a floor must be a value the
    population actually contains, or the band below it is not a band of rows."""

    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def decile_table(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Ten buckets over P, each with the counts §4.3 would have needed.

    ``known_windows`` is reported per decile and is the column that decides
    §4.3 condition 1. It is computed here rather than assumed so the artifact
    carries its own zero rather than citing one.
    """

    scored = [row for row in rows if row["score"] is not None]
    ordered = sorted(row["score"] for row in scored)
    boundaries = [
        {"decile": tenth, "fraction": tenth / 10.0, "value": _quantile(ordered, tenth / 10.0)}
        for tenth in range(1, 10)
    ]
    buckets: list[dict[str, Any]] = []
    edges = [ordered[0] if ordered else 0.0] + [edge["value"] for edge in boundaries] + [
        ordered[-1] if ordered else 0.0
    ]
    for tenth in range(10):
        low, high = edges[tenth], edges[tenth + 1]
        if tenth == 9:
            members = [row for row in scored if row["score"] >= low]
        else:
            members = [row for row in scored if low <= row["score"] < high]
        buckets.append(
            {
                "decile": tenth + 1,
                "range": [low, high],
                "n": len(members),
                "cold": sum(1 for row in members if row["cold"]),
                "admitted": sum(1 for row in members if row["admitted"]),
                "known_windows": 0,
                "exogenous_follow_rate": None,
                "wilson_95": [0.0, 1.0] if members else None,
            }
        )
    return {
        "population": len(scored),
        "unscored_rows": len(rows) - len(scored),
        "exactly_zero": sum(1 for row in scored if row["score"] == 0.0),
        "min": ordered[0] if ordered else None,
        "max": ordered[-1] if ordered else None,
        "mean": round(statistics.fmean(ordered), 6) if ordered else None,
        "boundaries": boundaries,
        "buckets": buckets,
        "distinct_positive_boundaries": sorted(
            {edge["value"] for edge in boundaries if edge["value"] > 0.0}
        ),
    }


def evaluate_rule(rows: Sequence[dict[str, Any]], table: dict[str, Any]) -> dict[str, Any]:
    """Apply S1/S2/S3 of §7.4 to every candidate boundary, highest first."""

    scored = [row for row in rows if row["score"] is not None]
    candidates_n = len(scored)
    admitted_rows = [row for row in scored if row["admitted"]]
    admitted_n = len(admitted_rows)

    verdicts: list[dict[str, Any]] = []
    for edge in sorted(table["boundaries"], key=lambda item: -item["value"]):
        threshold = edge["value"]
        removed = [row for row in scored if row["score"] < threshold]
        retained = [row for row in scored if row["score"] >= threshold]
        removed_admitted = [row for row in admitted_rows if row["score"] < threshold]

        s1 = threshold > 0.0 and bool(removed)
        share_candidates = len(removed) / candidates_n if candidates_n else 0.0
        share_admitted = len(removed_admitted) / admitted_n if admitted_n else 0.0
        s2 = share_candidates <= BUDGET_SHARE and share_admitted <= BUDGET_SHARE
        cold_removed = (
            sum(1 for row in removed if row["cold"]) / len(removed) if removed else 0.0
        )
        cold_retained = (
            sum(1 for row in retained if row["cold"]) / len(retained) if retained else 0.0
        )
        s3 = cold_removed <= cold_retained
        verdicts.append(
            {
                "decile": edge["decile"],
                "threshold": threshold,
                "S1_not_inert": s1,
                "S2_budget": s2,
                "S2_removed_share_candidates": round(share_candidates, 6),
                "S2_removed_share_admitted": round(share_admitted, 6),
                "S2_removed_candidates": len(removed),
                "S2_removed_admitted": len(removed_admitted),
                "S3_cold_non_regression": s3,
                "S3_cold_share_removed": round(cold_removed, 6),
                "S3_cold_share_retained": round(cold_retained, 6),
                "passes": bool(s1 and s2 and s3),
            }
        )
    licensed = next((item for item in verdicts if item["passes"]), None)
    return {
        "budget_share": BUDGET_SHARE,
        "candidates_denominator": candidates_n,
        "admitted_denominator": admitted_n,
        "per_boundary": verdicts,
        "licensed_threshold": licensed["threshold"] if licensed else None,
        "licensed_by": licensed,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", help="frozen snapshot (never the live db)")
    source.add_argument("--source-db", help="database to freeze a snapshot from")
    parser.add_argument(
        "--as-of",
        required=True,
        help="ISO instant; pins the query cut and every history maturity boundary",
    )
    parser.add_argument("--out", help="write the JSON report here (default: stdout)")
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--falsify",
        type=float,
        default=None,
        help="also replay the same query set with the valve charged at this "
        "threshold and report the coverage comparison of §7.6",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr, flush=True)

    if pool_usefulness_floor_from_env() is not None:
        raise SystemExit(
            "LM_MAP_POOL_USEFULNESS_GATE is armed in this process's environment. "
            "The census must measure the ungated population; unset it and re-run."
        )

    started = time.time()
    with frozen_snapshot(args.snapshot, args.source_db) as (snapshot, manifest):
        with working_copy(snapshot) as working:
            connection = sqlite3.connect(f"file:{working}?mode=ro", uri=True)
            events = _distinct_events(
                connection, limit=args.queries, seed=args.seed, as_of=args.as_of
            )
            connection.close()
            log(f"replaying {len(events)} distinct queries, as-of {args.as_of}")

            store = MemoryStore(str(working))
            service = MemoryRecallService(store)
            off = replay(
                service,
                store,
                events,
                as_of=args.as_of,
                max_results=args.max_results,
                collect_pool=True,
                log=log,
            )
            rows = off.pop("rows")
            table = decile_table(rows)
            rule = evaluate_rule(rows, table)

            falsifier: dict[str, Any] | None = None
            if args.falsify is not None:
                log(f"falsifier arm: valve charged at {args.falsify}")
                os.environ["LM_MAP_POOL_USEFULNESS_GATE"] = "1"
                os.environ["LM_MAP_POOL_MIN_USEFULNESS"] = repr(args.falsify)
                try:
                    on = replay(
                        service,
                        store,
                        events,
                        as_of=args.as_of,
                        max_results=args.max_results,
                        collect_pool=False,
                        log=log,
                    )
                finally:
                    os.environ.pop("LM_MAP_POOL_USEFULNESS_GATE", None)
                    os.environ.pop("LM_MAP_POOL_MIN_USEFULNESS", None)
                on.pop("rows", None)
                falsifier = {
                    "threshold": args.falsify,
                    "off": {
                        "covered": off["covered"],
                        "maps_with_clusters": off["maps_with_clusters"],
                        "admitted": off["admitted"],
                    },
                    "on": {
                        "covered": on["covered"],
                        "maps_with_clusters": on["maps_with_clusters"],
                        "admitted": on["admitted"],
                    },
                    "band": "one-sided: covered_on >= covered_off and maps_with_clusters_on >= maps_with_clusters_off",
                    "covered_holds": on["covered"] >= off["covered"],
                    "maps_hold": on["maps_with_clusters"] >= off["maps_with_clusters"],
                    "passes": on["covered"] >= off["covered"]
                    and on["maps_with_clusters"] >= off["maps_with_clusters"],
                }

    report = {
        "tool": "scripts/recall_map_pool_usefulness_census.py",
        "as_of": args.as_of,
        "snapshot_manifest": manifest,
        "replay": {
            "queries_requested": args.queries,
            "queries_replayed": len(events),
            "max_results": args.max_results,
            "seed": args.seed,
            "note": "max_results is the production recall cut (server.py:1040); the residual is everything past it",
        },
        "arm_off": off,
        "distribution": table,
        "rule": rule,
        "falsifier": falsifier,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
        log(f"wrote {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
