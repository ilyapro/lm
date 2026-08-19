#!/usr/bin/env python3
"""Run the whole post-session extraction stage over finished sessions.

    # propose only (the default is --dry-run): what WOULD be written, and why
    scripts/post_session_extract.py --split eval --max-sessions 8 --out report.json

    # one transcript, actually writing, against the local unit
    scripts/post_session_extract.py --transcript ~/.claude/projects/x/y.jsonl \
        --write --out report.json

Flow: transcript path (or session key, or a corpus split) -> SessionRecord ->
proposed remember ops (``postsession.insights``) + teach ops
(``postsession.corrections``) + attestation evidence (``postsession.evidence``)
-> gate check -> writes through the live Living Memory MCP interface -> run
report.

The consumption gate is checked FIRST, before any session is read: the
pre-registered bar in ``artifacts/post-session/usage-baseline.json`` names the
runner as its enforcer, and a verdict that fails, is missing, or is stale means
exit 3 with zero writes — an extractor whose traces are not recalled must
signal and stop, not keep writing.

Writing requires an explicit ``--write``; ``--dry-run`` is the default.
``--split holdout`` additionally requires ``--i-am-the-sealed-measurement-run``
so no ordinary run can burn the seal. Everything written goes over MCP to
``--url`` (default the local unit, ``http://127.0.0.1:8765/mcp/``) with the
bearer token from ``--token``, then ``$LM_AUTH_TOKEN``, then
``~/.config/living-memory/env`` — the SQLite file is never opened here.

Exit codes::

    0  ran to completion (a dry run counts)
    1  ran, but at least one accepted op failed to write
    2  usage or environment error (bad arguments, unreadable corpus, no seal)
    3  the consumption gate did not authorize the run; NOTHING was written
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from living_memory.postsession.client import (  # noqa: E402
    DEFAULT_ENV_FILE,
    DEFAULT_URL,
    ClientError,
    LiveMemoryClient,
    resolve_token,
)
from living_memory.postsession.corpus import (  # noqa: E402
    DEFAULT_MANIFEST,
    SPLITS,
    SessionEntry,
    index_path_for,
    iter_index,
    load_session,
)
from living_memory.postsession.corrections import (  # noqa: E402
    mask_reserved_splits,
    reserved_splits,
)
from living_memory.postsession.insights import ExtractionConfig  # noqa: E402
from living_memory.postsession.judge import (  # noqa: E402
    CassetteJudge,
    ClaudeCliJudge,
    FakeJudge,
    canonical_json,
    default_redactor,
)
from living_memory.postsession.runner import (  # noqa: E402
    DEFAULT_BASELINE,
    DEFAULT_BOOTSTRAP_ALLOWANCE,
    DEFAULT_STATE_DIR,
    DEFAULT_VERDICT,
    EXIT_GATE,
    EXIT_OK,
    EXIT_USAGE,
    ExtractionRunner,
    Ledger,
    RunnerConfig,
    build_report,
    evaluate_gate,
)
from living_memory.postsession.session import SOURCES, SessionRecord  # noqa: E402
from living_memory.postsession.transcripts import (  # noqa: E402
    TranscriptRef,
    load_transcript,
)

#: The CLI a bare transcript of each source was produced by.
_SOURCE_CLI = {
    "claude": "claude",
    "codex": "codex",
    "gigacode": "gigacode",
    "deepseek": "deepseek",
    "ae_chat": "ae",
    "ae_node_result": "ae",
}


def _sample_rank(session_key: str) -> str:
    """Deterministic session order: the same budget always picks the same set."""

    return hashlib.sha256(session_key.encode("utf-8")).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1], prog="post_session_extract.py"
    )
    what = parser.add_argument_group("what to extract from (pick exactly one)")
    what.add_argument("--transcript", type=Path, help="path to one transcript file")
    what.add_argument(
        "--source",
        choices=SOURCES,
        default="claude",
        help="transcript format for --transcript (default: %(default)s)",
    )
    what.add_argument(
        "--session",
        action="append",
        default=None,
        help="session key to look up in the corpus index (repeatable)",
    )
    what.add_argument("--split", choices=SPLITS, help="run over one corpus split")

    corpus = parser.add_argument_group("corpus")
    corpus.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    corpus.add_argument("--index", type=Path, default=None)
    corpus.add_argument("--max-sessions", type=int, default=8)
    corpus.add_argument("--max-bytes", type=int, default=12_000_000)

    gate = parser.add_argument_group("consumption gate")
    gate.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    gate.add_argument(
        "--gate-verdict",
        type=Path,
        default=DEFAULT_VERDICT,
        help="scripts/counterfactual_consumption.py report (default: %(default)s)",
    )
    gate.add_argument(
        "--gate-control",
        type=Path,
        default=None,
        help="organic control report, required once the verdict measures an "
        "extracted cohort",
    )
    gate.add_argument("--max-verdict-age-days", type=float, default=14.0)
    gate.add_argument(
        "--bootstrap-write-allowance",
        type=int,
        default=DEFAULT_BOOTSTRAP_ALLOWANCE,
        help="nodes a control-only verdict may ever authorize (default: %(default)s)",
    )

    write = parser.add_argument_group("writing")
    write.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="propose and decide, write nothing (the default)",
    )
    write.add_argument(
        "--write",
        action="store_true",
        help="actually execute accepted ops over MCP; without this flag the run "
        "is a dry run",
    )
    write.add_argument(
        "--i-am-the-sealed-measurement-run",
        action="store_true",
        help="required with --split holdout; an ordinary run must not burn the seal",
    )
    write.add_argument("--url", default=DEFAULT_URL)
    write.add_argument("--token", default=None)
    write.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    write.add_argument("--timeout", type=float, default=30.0)
    write.add_argument(
        "--lm-dedup",
        action="store_true",
        help="also consult the live server for near-duplicates during a dry run "
        "(a write run always does)",
    )
    write.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help="idempotency ledger directory (default: %(default)s)",
    )

    budget = parser.add_argument_group("budget")
    budget.add_argument("--max-ops-per-session", type=int, default=3)
    budget.add_argument("--max-total-ops", type=int, default=24)
    budget.add_argument("--max-attests-per-session", type=int, default=8)
    budget.add_argument("--max-total-attests", type=int, default=64)
    budget.add_argument("--max-judge-calls", type=int, default=12)

    judge = parser.add_argument_group("judge")
    judge.add_argument(
        "--judge", default="claude", choices=("claude", "cassette", "fake", "none")
    )
    judge.add_argument("--model", default="sonnet")
    judge.add_argument("--cassette", type=Path, default=None)
    judge.add_argument("--workers", type=int, default=4)

    parser.add_argument("--stages", default="insights,corrections,attest")
    parser.add_argument("--out", type=Path, default=None, help="run report path")
    parser.add_argument("--progress", action="store_true")
    return parser


def make_judge_factory(args: argparse.Namespace):
    """A fresh judge per stage call, so spend is attributed cleanly."""

    if args.judge == "none":
        return None
    if args.judge == "claude":
        return lambda: ClaudeCliJudge(model=args.model)
    if args.judge == "cassette":
        path = args.cassette or Path("tests/fixtures/postsession/cassettes/runner.json")
        return lambda: CassetteJudge(path, inner=ClaudeCliJudge(model=args.model), record=True)
    if args.judge == "fake":
        return lambda: FakeJudge([{"verdict": "none", "reason": "fake judge"}] * 10_000)
    raise SystemExit(f"unknown judge backend: {args.judge}")


def select_records(
    args: argparse.Namespace, log,
) -> tuple[list[SessionRecord], list[str], dict[str, Any]]:
    """The sessions this run reads, plus the splits it read and the accounting."""

    notes: dict[str, Any] = {}
    if args.transcript:
        path = args.transcript.expanduser()
        if not path.is_file():
            raise SystemExit(f"transcript not found: {path}")
        ref = TranscriptRef(
            source=args.source,
            cli=_SOURCE_CLI.get(args.source, "unknown"),
            path=path,
            session_key=f"{args.source}:{path.stem}",
            cli_session_id=path.stem,
        )
        return [load_transcript(ref)], [], notes

    index_path = args.index or index_path_for(args.manifest)
    if not Path(index_path).is_file():
        raise SystemExit(
            f"corpus index not found: {index_path}\n"
            "build it with: scripts/postsession_corpus.py --build"
        )
    entries: list[SessionEntry]
    if args.session:
        wanted = set(args.session)
        entries = [e for e in iter_index(Path(index_path)) if e.session_key in wanted]
        missing = wanted - {e.session_key for e in entries}
        if missing:
            raise SystemExit(
                f"session key(s) not in the corpus index: {', '.join(sorted(missing))}"
            )
        splits_read = sorted({e.split for e in entries})
        notes["session_keys"] = sorted(wanted)
        sealed = sorted(
            e.session_key for e in entries if e.split in reserved_splits()
        )
        if sealed and not args.i_am_the_sealed_measurement_run:
            raise SystemExit(
                f"session(s) in the sealed measurement split: {', '.join(sealed)}; "
                "an ordinary run may not read them"
            )
    else:
        entries = [e for e in iter_index(Path(index_path)) if e.split == args.split]
        splits_read = [args.split]
        entries.sort(key=lambda e: _sample_rank(e.session_key))
        oversized = [e for e in entries if e.bytes > args.max_bytes]
        entries = [e for e in entries if e.bytes <= args.max_bytes]
        notes["sessions_in_split"] = len(entries) + len(oversized)
        notes["sessions_too_large"] = len(oversized)
        if args.max_sessions:
            notes["sessions_beyond_max_sessions"] = max(
                0, len(entries) - args.max_sessions
            )
            entries = entries[: args.max_sessions]
    records: list[SessionRecord] = []
    failures = 0
    for entry in entries:
        try:
            records.append(load_session(entry))
        except Exception as exc:  # noqa: BLE001 - a corrupt transcript is not fatal
            failures += 1
            log(f"! load failed {entry.session_key}: {exc}")
    notes["sessions_selected"] = len(entries)
    notes["sessions_loaded"] = len(records)
    notes["sessions_failed_to_load"] = failures
    return records, splits_read, notes


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    chosen = [bool(args.transcript), bool(args.session), bool(args.split)]
    if sum(chosen) != 1:
        parser.print_usage(sys.stderr)
        log("pick exactly one of --transcript, --session, --split")
        return EXIT_USAGE

    # The seal, before anything is opened.
    if args.split in reserved_splits() and not args.i_am_the_sealed_measurement_run:
        log(
            f"--split {args.split} is the sealed measurement split; an ordinary "
            "run may not read it. Pass --i-am-the-sealed-measurement-run only "
            "from the field-measurement procedure."
        )
        return EXIT_USAGE

    dry_run = not args.write
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    ledger = Ledger(args.state_dir.expanduser())

    # The gate, before any session is read or any judge is paid.
    gate_decision = evaluate_gate(
        args.baseline,
        args.gate_verdict,
        args.gate_control,
        max_age_days=args.max_verdict_age_days,
        written_nodes_total=ledger.written_nodes_total(),
        bootstrap_allowance=args.bootstrap_write_allowance,
    )
    if not gate_decision.authorized:
        log(f"GATE {gate_decision.status}: this run is not authorized to write.")
        for reason in gate_decision.reasons:
            log(f"  - {reason}")
        log("Nothing was written. (exit 3)")
        if args.out:
            _write_report(
                args.out,
                build_report(
                    mode="halted",
                    splits_read=[],
                    gate_decision=gate_decision,
                    proposals=[],
                    decisions=[],
                    config=RunnerConfig(dry_run=dry_run, stages=stages),
                    ledger=ledger,
                ),
                splits_read=[],
            )
        return EXIT_GATE
    log(
        f"GATE PASS ({gate_decision.mode}): "
        + "; ".join(gate_decision.reasons or ("bar cleared",))
    )

    try:
        records, splits_read, notes = select_records(args, log)
    except SystemExit as exc:
        log(str(exc))
        return EXIT_USAGE

    client: LiveMemoryClient | None = None
    if args.write or args.lm_dedup:
        token = resolve_token(args.token, env_file=args.env_file)
        client = LiveMemoryClient(args.url, token=token, timeout=args.timeout)
        try:
            client.ping()
        except ClientError as exc:
            log(f"cannot reach the Living Memory host: {exc}")
            return EXIT_USAGE

    config = RunnerConfig(
        stages=stages,
        dry_run=dry_run,
        max_ops_per_session=args.max_ops_per_session,
        max_total_ops=args.max_total_ops,
        max_attests_per_session=args.max_attests_per_session,
        max_total_attests=args.max_total_attests,
        insights=ExtractionConfig(max_judge_calls=args.max_judge_calls),
    )
    runner = ExtractionRunner(
        config=config,
        ledger=ledger,
        client=client,
        judge_factory=make_judge_factory(args),
    )

    if args.progress:
        log(f"proposing over {len(records)} session(s), stages={','.join(stages)}")
    if args.workers > 1 and len(records) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            proposals = list(pool.map(runner.propose, records))
    else:
        proposals = [runner.propose(record) for record in records]
    decisions = runner.decide_and_execute(proposals)

    report = build_report(
        mode="dry-run" if dry_run else "write",
        splits_read=splits_read,
        gate_decision=gate_decision,
        proposals=proposals,
        decisions=decisions,
        config=config,
        ledger=ledger,
        extra={"selection": notes},
    )
    if args.out:
        _write_report(args.out, report, splits_read=splits_read)
        log(f"wrote {args.out}")

    ops = report["ops"]
    attests = report["attestations"]
    log(
        f"ops: {ops['proposed']} proposed, "
        f"{ops['by_status'].get('written', 0)} written, "
        f"{ops['by_status'].get('would_write', 0)} would write, "
        f"{report['budgets']['dropped']} dropped over budget"
    )
    log(
        f"attestations: {attests['proposed']} proposed, "
        f"{attests['applied']} applied, {attests['grounded_results']} grounded results"
    )
    log(f"judge: {report['judge']['calls']} calls, ${report['judge']['total_cost_usd']}")
    failed = report["ops"]["by_status"].get("failed", 0) + report["attestations"][
        "by_status"
    ].get("failed", 0)
    return EXIT_OK if not failed else 1


def _write_report(path: Path, report: dict[str, Any], *, splits_read: Sequence[str]) -> None:
    """Redact, mask the sealed split's name (unless this IS the sealed run),
    and write the report."""

    text = default_redactor(canonical_json(report))
    if not any(split in reserved_splits() for split in splits_read):
        text = mask_reserved_splits(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
