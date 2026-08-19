#!/usr/bin/env python3
"""Counterfactual consumption gate: replay recorded recalls against a re-inserted cohort.

The retrospective metric (``scripts/trace_usage_linkage.py``) can only score
memory that already lived through a stretch of recall traffic. This script
scores a cohort the other way round: it freezes the database, removes the
cohort from a disposable working copy, puts it back through the store's own
write path, and then replays the recorded post-cutoff recall requests end to
end through the *current* retrieval code — measuring how often the cohort comes
back in the delivered results, plus the grounded variant.

    PYTHONPATH=src python3 scripts/counterfactual_consumption.py \
        --db ~/.local/share/living-memory/global.sqlite3 \
        --as-of 2026-08-19T00:00:00Z \
        --window 2026-08-12..2026-08-15 \
        --replay-since 2026-08-12 \
        --cohort-rule 'nodes.created_at in [2026-08-12,2026-08-15) UTC …' \
        --out artifacts/post-session/counterfactual-selfcheck.json

``--cohort-rule`` is mandatory and is the pre-registered selection rule in the
operator's own words: a counterfactual whose cohort was chosen after seeing the
answer proves nothing, so the rule is published next to the number it produced.

The live database is never opened writably. ``--db`` is read through ``mode=ro``
and copied with the SQLite backup API (a plain ``cp`` would lose the WAL, which
on the field database is half the data); ``MemoryStore`` — which migrates and
writes whatever file it opens — only ever sees the scratch working copy. Pass
``--snapshot`` to reuse an already frozen file and skip the copy.

The definitions all live in :mod:`living_memory.postsession.counterfactual` and
:mod:`living_memory.postsession.usage_metric`; this script is argument parsing,
the aggregate-only privacy guard and JSON output.
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
from living_memory.postsession.counterfactual import (  # noqa: E402
    REINSERT_MODES,
    CoherenceError,
    CounterfactualConfig,
    PrivacyGuardError,
    check_privacy,
    run_counterfactual,
)
from living_memory.postsession.usage_metric import (  # noqa: E402
    DEFAULT_DB_PATH,
    DEFAULT_WITHIN_DAYS,
    NULL_AGENT_TOKENS,
    SAFE_STRING,
    parse_context_key,
    parse_window,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH.expanduser()),
        help="source database (read-only; frozen with the SQLite backup API)",
    )
    parser.add_argument(
        "--snapshot",
        help="reuse this already-frozen snapshot instead of freezing --db again",
    )
    parser.add_argument(
        "--as-of",
        required=True,
        help="ISO instant; cuts BOTH nodes and recall_events on created_at <= as_of",
    )
    parser.add_argument(
        "--window",
        required=True,
        help="cohort creation window LO..HI, half-open [LO, HI); bare dates mean midnight UTC",
    )
    parser.add_argument(
        "--replay-since",
        required=True,
        help="replay every recorded recall event at or after this instant; pass the "
        "cohort window start for a COMPLETE replay (an earlier event can never consume "
        "a cohort node, so that covers every possible consumer)",
    )
    parser.add_argument(
        "--cohort-rule",
        required=True,
        help="the PRE-REGISTERED cohort selection rule, published with the result "
        "(printable ASCII, <= 200 chars)",
    )
    parser.add_argument(
        "--cohort-kind",
        default="organic_holdout",
        help="what kind of cohort this is (default: %(default)s)",
    )
    parser.add_argument(
        "--agent",
        help=f"restrict the cohort to nodes.agent=NAME ({'/'.join(NULL_AGENT_TOKENS)} "
        "selects agent IS NULL)",
    )
    parser.add_argument(
        "--context-key",
        metavar="K=V",
        help="restrict the cohort to nodes whose context JSON has top-level K=V "
        "(K= with an empty value means the key is present)",
    )
    parser.add_argument(
        "--reinsert-mode",
        choices=REINSERT_MODES,
        default="faithful",
        help="faithful restores the node exactly (the organic control); fresh re-inserts "
        "with no accumulated history, as a candidate trace would arrive (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-partial-replay",
        action="store_true",
        help="accept a --replay-since later than the cohort window start, which leaves "
        "recorded consumptions the replay was never given a chance to reproduce",
    )
    parser.add_argument(
        "--diagnostic-replay-since",
        metavar="ISO",
        help="also run the whole protocol from this second replay window and publish it "
        "under diagnostics.alternate_replay, so the gate's dependence on which recorded "
        "traffic it is handed is visible in the artifact rather than only to whoever ran it",
    )
    parser.add_argument(
        "--fresh-arm",
        action="store_true",
        help="also run the fresh-history arm on its own working copy, to price what a "
        "node's accumulated history is worth",
    )
    parser.add_argument(
        "--min-containment",
        type=float,
        default=DEFAULT_MIN_CONTAINMENT,
        help="IDF containment at which a closing trace grounds a node (default: %(default)s)",
    )
    parser.add_argument(
        "--within-days",
        type=int,
        default=DEFAULT_WITHIN_DAYS,
        help="'recalled again soon' horizon in days (default: %(default)s)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        help="cap the replay at N requests (smoke runs only; the report says so)",
    )
    parser.add_argument("--out", metavar="PATH", help="write the JSON report here (default: stdout)")
    parser.add_argument("--quiet", action="store_true", help="suppress progress on stderr")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not SAFE_STRING.match(args.cohort_rule):
        raise SystemExit(
            "--cohort-rule must be printable ASCII of at most 200 characters: it is "
            "published in an aggregates-only artifact"
        )

    db_path = Path(args.db).expanduser()
    snapshot = Path(args.snapshot).expanduser() if args.snapshot else None
    if snapshot is not None and not snapshot.exists():
        raise SystemExit(f"snapshot not found: {snapshot}")
    if snapshot is None and not db_path.exists():
        raise SystemExit(f"database not found: {db_path}")

    try:
        window = parse_window(args.window)
        context_key = parse_context_key(args.context_key) if args.context_key else None
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    agent = args.agent
    agent_is_null = agent in NULL_AGENT_TOKENS if agent else False

    config = CounterfactualConfig(
        window=window,
        replay_since=args.replay_since,
        as_of=args.as_of,
        cohort_rule=args.cohort_rule,
        cohort_kind=args.cohort_kind,
        agent=None if agent_is_null else agent,
        agent_is_null=agent_is_null,
        context_key=context_key,
        reinsert_mode=args.reinsert_mode,
        also_fresh_arm=args.fresh_arm,
        allow_partial_replay=args.allow_partial_replay,
        diagnostic_replay_since=args.diagnostic_replay_since,
        min_containment=args.min_containment,
        within_days=args.within_days,
        max_requests=args.max_requests,
    )

    def log(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr, flush=True)

    try:
        report = run_counterfactual(
            None if snapshot is not None else db_path,
            config,
            snapshot=snapshot,
            log=log,
        )
    except PrivacyGuardError as exc:
        raise SystemExit(str(exc)) from exc
    except CoherenceError as exc:
        raise SystemExit(f"working copy lost coherence, refusing to report: {exc}") from exc
    except (ValueError, OSError, sqlite3.Error) as exc:
        raise SystemExit(f"counterfactual failed: {exc}") from exc

    check_privacy(report)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(payload)

    check = report["selfcheck"]
    grounded = check["grounded"]
    print(
        "cohort=%s nodes=%s | retrospective=%s counterfactual=%s gap=%s (bar %s) -> %s"
        % (
            check["cohort_kind"],
            check["cohort_nodes"],
            check["retrospective_rate"],
            check["counterfactual_rate"],
            check["gap"],
            check["max_gap"],
            "PASS" if check["passed"] else "FAIL",
        )
    )
    print(
        "grounded: retrospective=%s/%s counterfactual=%s/%s gap=%s"
        % (
            grounded["retrospective_consumed_later"],
            grounded["retrospective_denominator"],
            grounded["counterfactual_consumed_later"],
            grounded["counterfactual_denominator"],
            grounded["gap"],
        )
    )
    print(
        "replay: %s recorded event(s) -> %s request(s), complete=%s, reproduced %s of "
        "recorded deliveries"
        % (
            report["protocol"]["request_stats"]["events_replayed"],
            report["protocol"]["request_stats"]["requests"],
            report["protocol"]["replay_is_complete"],
            report["arms"][f"counterfactual_{check['reinsert_mode']}"]["replay"]["fidelity"][
                "reproduced_share_of_recorded"
            ],
        )
    )
    alternate = report["diagnostics"]["alternate_replay"]
    if alternate is not None:
        print(
            "alternate replay from %s: retrospective=%s counterfactual=%s gap=%s -> %s"
            % (
                alternate["replay_since"],
                alternate["retrospective_rate"],
                alternate["counterfactual_rate"],
                alternate["gap"],
                "within bar" if alternate["within_max_gap"] else "OUTSIDE bar",
            )
        )
    agreement = check["agreement"]
    print(
        "per-node agreement: both=%s retro_only=%s counter_only=%s jaccard=%s"
        % (
            agreement["both"],
            agreement["retrospective_only"],
            agreement["counterfactual_only"],
            agreement["jaccard"],
        )
    )
    return 0 if check["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
