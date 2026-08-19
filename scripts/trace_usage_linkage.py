#!/usr/bin/env python3
"""Measure, read-only, whether the memory written in a window was recalled later.

This is the retrospective half of the extraction quality gate: it joins each
node to the recall events that delivered it *afterwards*, so "the extractor
wrote 300 traces" can be answered with "and 41% of them were recalled again",
which is the only number that falsifies a writer.

    PYTHONPATH=src python3 scripts/trace_usage_linkage.py \
        --db ~/.local/share/living-memory/global.sqlite3 \
        --as-of 2026-08-19T00:00:00Z \
        --window 2026-07-01..2026-08-01 \
        --agent ae --agent unattributed \
        --out artifacts/post-session/usage-july.json

``--as-of`` is required, not optional: the live database grows continuously and
a recall event recorded after a window closed can only add consumption to it, so
a run without a pinned cutoff is not reproducible and must not be published.

The definition, the cohort rule and the grounded variant's denominator all live
in :mod:`living_memory.postsession.usage_metric`; this script is argument
parsing, the aggregate-only privacy guard and JSON output. The database is
opened strictly read-only (``mode=ro``) — the live server is never touched.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from living_memory.grounding import DEFAULT_MIN_CONTAINMENT  # noqa: E402
from living_memory.postsession.usage_metric import (  # noqa: E402
    DEFAULT_DB_PATH,
    DEFAULT_WITHIN_DAYS,
    PrivacyGuardError,
    check_privacy,
    parse_context_key,
    run_metric,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH.expanduser()),
        help="database to read (opened read-only; never written)",
    )
    parser.add_argument(
        "--as-of",
        required=True,
        help="ISO instant; cuts BOTH nodes and recall_events on created_at <= as_of",
    )
    parser.add_argument(
        "--window",
        required=True,
        help="node creation window LO..HI, half-open [LO, HI); bare dates mean midnight UTC",
    )
    parser.add_argument(
        "--agent",
        action="append",
        default=[],
        metavar="NAME",
        help="repeatable: add a cohort restricted to nodes.agent=NAME "
        "(NAME 'unattributed' or 'NULL' selects agent IS NULL)",
    )
    parser.add_argument(
        "--context-key",
        action="append",
        default=[],
        metavar="K=V",
        help="repeatable: add a cohort whose nodes.context JSON has top-level K=V "
        "(K= with an empty value means the key is present)",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="write the JSON report here (default: stdout)",
    )
    parser.add_argument(
        "--within-days",
        type=int,
        default=DEFAULT_WITHIN_DAYS,
        help="'recalled again soon' horizon in days (default: %(default)s)",
    )
    parser.add_argument(
        "--min-containment",
        type=float,
        default=DEFAULT_MIN_CONTAINMENT,
        help="IDF containment at which a closing trace grounds a node "
        "(default: %(default)s, the shared live-path threshold)",
    )
    parser.add_argument(
        "--no-grounded",
        action="store_true",
        help="skip the grounded variant (raw consumption only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        raise SystemExit(f"database not found: {db_path}")

    try:
        context_keys = [parse_context_key(raw) for raw in args.context_key]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        report = run_metric(
            db_path,
            window=args.window,
            as_of=args.as_of,
            agents=args.agent,
            context_keys=context_keys,
            within_days=args.within_days,
            min_containment=args.min_containment,
            grounded=not args.no_grounded,
        )
    except PrivacyGuardError as exc:
        raise SystemExit(str(exc)) from exc
    except (ValueError, OSError, sqlite3.Error) as exc:
        raise SystemExit(f"measurement failed: {exc}") from exc

    check_privacy(report)
    payload = json.dumps(report, indent=2, sort_keys=True)

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(payload)

    for cohort in report["cohorts"]:
        grounded = cohort["grounded"]
        print(
            "%-28s nodes=%s consumed_later=%s rate=%s within_%sd=%s | "
            "grounded=%s/%s rate=%s"
            % (
                cohort["id"],
                cohort["nodes"],
                cohort["consumed_later"],
                cohort["rate"],
                cohort["within_days"],
                cohort["consumed_within_7d"],
                grounded["consumed_later"],
                grounded["nodes_with_closed_consumer"],
                grounded["rate"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
