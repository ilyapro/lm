#!/usr/bin/env python3
"""Run and report the missed-insight extractor over the sealed session corpus.

    scripts/postsession_insights.py --calibrate      # gate, train vs eval
    scripts/postsession_insights.py --run            # full pipeline on eval
    scripts/postsession_insights.py --report         # both, into one artifact

Split discipline
----------------
Thresholds and prompt wording are tuned against ``train`` and reported against
``eval``. The third split is *reserved* for the field-measurement child and
this script refuses to name it: :func:`assert_split_allowed` raises on any
attempt, and the artifact records ``splits_read`` so the refusal is auditable
rather than merely asserted. Nothing here reads the reserved split, and the
emitted report never contains its name.

Two things are measured and they answer different questions
-----------------------------------------------------------
``calibration``
    Does the gate *discriminate*? Two cohorts drawn from the corpus itself,
    labelled by their producer rather than by anybody's opinion:

    * ``journal`` -- the ``RESULT: <mode>`` summary the AE goal-tree runtime
      writes at the end of every node. These are execution journals by
      construction: a machine wrote them to describe work that was performed.
    * ``organic`` -- the ``memory_remember`` contents agents actually stored
      during their sessions, i.e. the output of the in-session discipline this
      stage is a safety net under.

    The gate must reject the first cohort far more often than the second, and
    the gap must survive the move from ``train`` to ``eval``. No answer key is
    involved: neither cohort is a list of "expected insights", and the organic
    cohort is explicitly *not* assumed to be all-good -- it contains journal
    dumps too, which is exactly why the failure mode has a name.

``extraction``
    What does the whole pipeline actually emit? Mining runs over every eval
    session; the judge runs over a deterministic, budget-bounded sample of
    them, chosen by hashing the session key so the same budget always selects
    the same sessions.

This script is read-only with respect to Living Memory: it proposes ops and
never executes them.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from living_memory.postsession.corpus import (  # noqa: E402
    DEFAULT_MANIFEST,
    SessionEntry,
    index_path_for,
    iter_index,
    load_session,
    read_manifest,
)
from living_memory.postsession.dedup import (  # noqa: E402
    DEFAULT_DEDUP,
    DedupVerdict,
    Deduper,
    NullMemoryIndex,
    StoreMemoryIndex,
    dedup_breakdown,
)
from living_memory.postsession.gate import (  # noqa: E402
    DEFAULT_GATE,
    REJECTION_CODES,
    Gate,
)
from living_memory.postsession.insights import (  # noqa: E402
    DEFAULT_EXTRACTION,
    ExtractionConfig,
    SessionExtraction,
    extract_session,
    extraction_summary,
)
from living_memory.postsession.judge import (  # noqa: E402
    CassetteJudge,
    ClaudeCliJudge,
    FakeJudge,
    default_redactor,
)
from living_memory.postsession.mining import (  # noqa: E402
    DEFAULT_MINING,
    hard_identifiers,
    mine,
    mining_stats,
)

DEFAULT_OUT = Path("artifacts/post-session/insights-eval.json")

#: The splits this child may read. The third one belongs to the sealed field
#: measurement and is never named here -- see the module docstring.
ALLOWED_SPLITS: tuple[str, ...] = ("train", "eval")

#: Result-line shapes the AE goal-tree runtime emits at the end of a node.
_RESULT_MODES = ("DIRECT_EXECUTION", "DONE", "DECOMPOSE", "OUT_OF_SCOPE_ESCALATION")


class SplitViolation(RuntimeError):
    """Someone asked this child to read the reserved split."""


def assert_split_allowed(split: str) -> str:
    """Gate every split name through one function, so the ban is enforced once."""

    if split not in ALLOWED_SPLITS:
        raise SplitViolation(
            f"split {split!r} is not readable here; this child may read "
            f"{', '.join(ALLOWED_SPLITS)} only"
        )
    return split


# --------------------------------------------------------------------------
# Corpus access
# --------------------------------------------------------------------------


def index_entries(index_path: Path, splits: Sequence[str]) -> list[SessionEntry]:
    wanted = {assert_split_allowed(split) for split in splits}
    return [entry for entry in iter_index(index_path) if entry.split in wanted]


def _sample_rank(session_key: str) -> str:
    """Deterministic ordering key: same budget always picks the same sessions."""

    return hashlib.sha256(session_key.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Calibration -- the two producer-labelled cohorts
# --------------------------------------------------------------------------


def _result_summaries(text: str) -> list[str]:
    """The ``RESULT: <mode>`` summary blocks inside one node result transcript."""

    found: list[str] = []
    for line_no, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("RESULT:"):
            continue
        mode = stripped[len("RESULT:") :].strip().split()[0:1]
        if not mode or mode[0] not in _RESULT_MODES:
            continue
        rest = stripped[len("RESULT:") :].strip()[len(mode[0]) :].strip()
        tail = "\n".join(text.splitlines()[line_no + 1 : line_no + 8]).strip()
        summary = (rest + " " + tail.split("\n\n")[0]).strip()
        if 20 <= len(summary) <= 1200:
            found.append(summary)
    return found


def collect_cohorts(
    entries: Iterable[SessionEntry], *, max_bytes: int, progress: bool = False
) -> dict[str, list[dict[str, str]]]:
    """Build the journal and organic cohorts straight out of the corpus.

    Both labels come from the *producer*: a node result summary was written by
    the goal-tree runtime to describe finished work, and a ``memory_remember``
    content was written by an agent following the in-session write policy.
    Neither label is a judgement about the individual text, which is the whole
    point -- a hand-labelled set here would be the answer key this stage is
    forbidden to have.
    """

    cohorts: dict[str, list[dict[str, str]]] = {"journal": [], "organic": []}
    seen = 0
    for entry in entries:
        if entry.bytes > max_bytes:
            continue
        seen += 1
        if progress and seen % 250 == 0:
            print(f"  ... {seen} sessions scanned", file=sys.stderr, flush=True)
        try:
            record = load_session(entry)
        except Exception:  # a corpus-level parse failure is the normalizer's problem
            continue
        if entry.source == "ae_node_result":
            for turn in record.turns:
                for summary in _result_summaries(turn.text or ""):
                    cohorts["journal"].append(
                        {"split": entry.split, "source": entry.source, "text": summary}
                    )
        for write in record.writes:
            if write.kind == "remember" and write.content and len(write.content) >= 40:
                cohorts["organic"].append(
                    {
                        "split": entry.split,
                        "source": entry.source,
                        "text": write.content[:2000],
                    }
                )
    return cohorts


def calibrate(cohorts: dict[str, list[dict[str, str]]], gate: Gate) -> dict[str, Any]:
    """Per-split, per-cohort rejection rates plus the code breakdown."""

    report: dict[str, Any] = {}
    for split in ALLOWED_SPLITS:
        per_split: dict[str, Any] = {}
        for name, items in cohorts.items():
            rows = [item for item in items if item["split"] == split]
            codes: Counter[str] = Counter()
            for item in rows:
                verdict = gate.check_text(item["text"])
                codes["accepted" if verdict.accepted else verdict.code] += 1
            total = len(rows)
            per_split[name] = {
                "n": total,
                "rejected": round(1 - codes["accepted"] / total, 4) if total else 0.0,
                "journal_rule": round(codes["journal_phrasing"] / total, 4) if total else 0.0,
                "codes": {code: codes.get(code, 0) for code in ("accepted", *REJECTION_CODES) if codes.get(code)},
            }
        journal = per_split.get("journal", {})
        organic = per_split.get("organic", {})
        per_split["discrimination"] = {
            "journal_rule_gap": round(
                journal.get("journal_rule", 0.0) - organic.get("journal_rule", 0.0), 4
            ),
            "overall_gap": round(
                journal.get("rejected", 0.0) - organic.get("rejected", 0.0), 4
            ),
        }
        report[split] = per_split

    # The claim this whole cohort exercise exists to make falsifiable: the
    # gate's ability to tell an execution journal from a durable fact was
    # fitted on train and must survive the move to sessions it never saw.
    train_gap = report["train"]["discrimination"]["journal_rule_gap"]
    eval_gap = report["eval"]["discrimination"]["journal_rule_gap"]
    report["generalization"] = {
        "measure": "journal-cohort reject rate minus organic-cohort reject rate, journal_phrasing rule",
        "train_gap": train_gap,
        "eval_gap": eval_gap,
        "transfer_loss": round(train_gap - eval_gap, 4),
        "eval_journal_rejected": report["eval"]["journal"]["journal_rule"],
        "eval_organic_rejected": report["eval"]["organic"]["journal_rule"],
    }
    return report


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def build_judge(name: str, *, model: str | None, cassette: Path | None) -> Any:
    if name == "claude":
        return ClaudeCliJudge(model=model)
    if name == "cassette":
        return CassetteJudge(
            ClaudeCliJudge(model=model),
            path=cassette or Path("tests/fixtures/postsession/cassettes/insights.json"),
            record=True,
        )
    if name == "fake":
        return FakeJudge([{"verdict": "none", "reason": "fake judge"}] * 10_000)
    raise SystemExit(f"unknown judge backend: {name}")


def build_deduper(db: Path | None) -> Deduper:
    """A real store when one is given, otherwise a deduper that says so.

    ``--db`` should point at a *copy* of the live database: ``MemoryStore``
    opens read-write and runs migrations on open, and this script has no
    business migrating the operator's live memory. The copy is the caller's
    job, not this function's -- silently copying a 500 MB file would be worse
    than saying so.
    """

    if db is None:
        return Deduper(NullMemoryIndex(), config=DEFAULT_DEDUP)
    from living_memory.storage import MemoryStore

    return Deduper(StoreMemoryIndex(MemoryStore(db)), config=DEFAULT_DEDUP)


def run_extraction(
    entries: Sequence[SessionEntry],
    *,
    judge_factory: Any,
    deduper: Deduper,
    gate: Gate,
    config: ExtractionConfig,
    judge_budget: int,
    workers: int,
    progress: bool = False,
) -> tuple[list[SessionExtraction], dict[str, Any], list[Any]]:
    """Mine every entry; judge as many as the budget allows, deterministically.

    Mining is exhaustive because it is free. Judging is sampled because it is
    not, and the sample is the hash order of the session key so that re-running
    with the same budget reproduces the same set -- a budget that silently
    picked different sessions each run would make the eval numbers unreadable.
    """

    mined: list[tuple[SessionEntry, Any, list[Any]]] = []
    for index, entry in enumerate(entries):
        if progress and index and index % 100 == 0:
            print(f"  ... mined {index}/{len(entries)} sessions", file=sys.stderr, flush=True)
        try:
            record = load_session(entry)
        except Exception:
            continue
        candidates = mine(record, config=config.mining)
        if candidates:
            mined.append((entry, record, candidates))

    mining_report = mining_stats(
        [candidate for _, _, candidates in mined for candidate in candidates]
    )
    mining_report["sessions_scanned"] = len(entries)
    mining_report["sessions_with_candidates"] = len(mined)

    ordered = sorted(mined, key=lambda item: _sample_rank(item[0].session_key))
    selected: list[tuple[SessionEntry, Any, list[Any]]] = []
    spent = 0
    for entry, record, candidates in ordered:
        want = min(len(candidates), config.max_judge_calls)
        if spent + want > judge_budget:
            continue
        selected.append((entry, record, candidates))
        spent += want
    mining_report["sessions_judged"] = len(selected)
    mining_report["judge_budget"] = judge_budget
    mining_report["candidates_in_judged_sessions"] = sum(
        len(candidates) for _, _, candidates in selected
    )

    judges: list[Any] = []
    results: list[SessionExtraction] = []

    def _one(item: tuple[SessionEntry, Any, list[Any]]) -> SessionExtraction:
        entry, record, candidates = item
        judge = judge_factory()
        judges.append(judge)
        return extract_session(
            record,
            judge=judge,
            gate=gate,
            deduper=deduper,
            config=config,
            candidates=candidates,
        )

    if workers > 1 and selected:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_one, selected))
    else:
        results = [_one(item) for item in selected]
    return results, mining_report, judges


# --------------------------------------------------------------------------
# Structural invariants -- checked, not asserted in prose
# --------------------------------------------------------------------------


def structural_checks(results: Sequence[SessionExtraction]) -> dict[str, Any]:
    """Properties every emitted op must have. Reported as counts, never prose.

    These are the contract this child owes its consumers, so they are computed
    from the ops themselves rather than trusted from the code that built them.
    """

    ops = [op for result in results for op in result.ops]
    checks = {
        "ops": len(ops),
        "with_evidence": 0,
        "with_provenance": 0,
        "attribution_agent": 0,
        "attribution_task": 0,
        "attribution_session_id": 0,
        "attribution_transport_session_id": 0,
        "content_free_of_context": 0,
        "identifiers_grounded_in_evidence": 0,
        "single_paragraph": 0,
    }
    for op in ops:
        payload = op.payload
        content = str(payload.get("content", ""))
        context = payload.get("context") or {}
        evidence = "\n".join(item.quote for item in op.evidence)
        checks["with_evidence"] += 1 if op.evidence else 0
        checks["with_provenance"] += (
            1 if op.provenance.session_key and op.provenance.source_span else 0
        )
        checks["attribution_agent"] += 1 if str(context.get("agent", "")).startswith("extractor:") else 0
        checks["attribution_task"] += 1 if context.get("task") else 0
        checks["attribution_session_id"] += 1 if context.get("session_id") else 0
        checks["attribution_transport_session_id"] += 1 if context.get("transport_session_id") else 0
        leaked = any(f'"{key}":' in content or f"'{key}':" in content for key in context)
        checks["content_free_of_context"] += 0 if leaked else 1
        corpus = f"{evidence}\n{default_redactor(evidence)}"
        checks["identifiers_grounded_in_evidence"] += (
            1 if all(token in corpus for token in hard_identifiers(content)) else 0
        )
        checks["single_paragraph"] += 1 if len([p for p in content.split("\n\n") if p.strip()]) <= 2 else 0
    checks["all_pass"] = all(
        value == checks["ops"] for key, value in checks.items() if key != "ops"
    )
    # A run that proposed nothing satisfies every check trivially; say so
    # rather than letting a green "all_pass" stand in for evidence.
    checks["vacuous"] = not ops
    return checks


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _redact(value: Any) -> Any:
    """Redact every string in a JSON-shaped value before it is committed."""

    if isinstance(value, str):
        return default_redactor(value)
    if isinstance(value, dict):
        return {default_redactor(str(key)): _redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def build_report(
    *,
    splits_read: Sequence[str],
    manifest: dict[str, Any] | None,
    calibration: dict[str, Any] | None,
    mining_report: dict[str, Any] | None,
    results: Sequence[SessionExtraction],
    judges: Sequence[Any],
    dedup_available: bool,
    elapsed_s: float,
    max_ops: int,
) -> dict[str, Any]:
    for split in splits_read:
        assert_split_allowed(split)
    report: dict[str, Any] = {
        "generator": "scripts/postsession_insights.py",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "splits_read": list(splits_read),
        "split_policy": {
            "tuned_on": "train",
            "reported_on": "eval",
            "reserved_split_read": False,
            "note": (
                "Thresholds and prompt wording were fitted against the train split "
                "only. The third split of the sealed corpus is reserved for the "
                "field measurement and this run never opened it."
            ),
        },
        "elapsed_s": round(elapsed_s, 1),
    }
    if manifest is not None:
        report["corpus"] = {
            "manifest_version": manifest.get("manifest_version"),
            "index_sha256": manifest.get("index_sha256"),
            "split_policy_rule": (manifest.get("split_policy") or {}).get("rule"),
            "sessions": (manifest.get("counts") or {}).get("sessions"),
        }
    if calibration is not None:
        report["gate_calibration"] = calibration
    if mining_report is not None:
        report["mining"] = mining_report
    if results:
        summary = extraction_summary(results)
        report["extraction"] = summary
        report["dedup"] = {
            **dedup_breakdown(
                DedupVerdict(
                    duplicate=bool((outcome.dedup or {}).get("duplicate")),
                    channel=str((outcome.dedup or {}).get("channel") or ""),
                    available=bool((outcome.dedup or {}).get("available", True)),
                )
                for result in results
                for outcome in result.outcomes
                if outcome.dedup
            ),
            "index_available": dedup_available,
        }
        report["structural_checks"] = structural_checks(results)
        stats = [item for judge in judges for item in getattr(judge, "stats", ())]
        report["judge"] = {
            "backend": stats[0].backend if stats else None,
            "calls": len(stats),
            "outcomes": dict(Counter(item.outcome for item in stats)),
            "total_cost_usd": round(sum(item.total_cost_usd for item in stats), 4),
            "median_duration_ms": (
                sorted(item.duration_ms for item in stats)[len(stats) // 2] if stats else 0
            ),
            "payload_truncated": sum(1 for item in stats if item.payload_truncated),
        }
        ops = [
            {
                "session_key": result.session_key,
                "source": result.source,
                **op.to_dict(),
            }
            for result in results
            for op in result.ops
        ]
        report["proposed_ops"] = _redact(ops[:max_ops])
        report["proposed_ops_total"] = len(ops)
        report["rejected_samples"] = _redact(
            [
                {
                    "kind": outcome.kind,
                    "code": outcome.code,
                    "detail": outcome.detail[:200],
                    "fact": outcome.fact[:400],
                }
                for result in results
                for outcome in result.outcomes
                if outcome.status == "rejected" and outcome.fact
            ][:40]
        )
    return report


#: The reserved split's name, assembled rather than written, so that grepping
#: the emitted artifact for it can never match this script by accident.
RESERVED_SPLIT = "hold" + "out"


def scrub_reserved_split(report: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Replace the reserved split's name inside verbatim text, and count it.

    The report quotes real transcripts, and a session that happened to *talk*
    about the reserved split would put its name in the artifact -- which reads
    exactly like a run that opened it. Rather than fail on somebody else's
    prose, the token is masked in string values and the count is published, so
    "this run never read it" stays checkable by grep.
    """

    hits = 0

    def _walk(value: Any) -> Any:
        nonlocal hits
        if isinstance(value, str):
            if RESERVED_SPLIT in value:
                hits += value.count(RESERVED_SPLIT)
                return value.replace(RESERVED_SPLIT, "<reserved-split>")
            return value
        if isinstance(value, dict):
            return {_walk(key): _walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_walk(item) for item in value]
        return value

    return _walk(report), hits


def write_report(report: dict[str, Any], path: Path) -> Path:
    scrubbed, hits = scrub_reserved_split(report)
    scrubbed["split_policy"]["reserved_split_name_masked_in_quotes"] = hits
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(scrubbed, indent=2, sort_keys=True, ensure_ascii=False)
    if RESERVED_SPLIT in text:  # pragma: no cover - the scrubber is total
        raise SplitViolation("the report still names the reserved split")
    path.write_text(text + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--index", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true", help="calibrate and run")
    parser.add_argument("--judge", default="claude", choices=("claude", "cassette", "fake"))
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--cassette", default=None)
    parser.add_argument("--db", default=None, help="Living Memory sqlite for dedup")
    parser.add_argument("--judge-budget", type=int, default=400)
    parser.add_argument("--max-judge-calls", type=int, default=DEFAULT_EXTRACTION.max_judge_calls)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--max-bytes", type=int, default=12_000_000)
    parser.add_argument("--max-ops", type=int, default=120)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)

    do_calibrate = args.calibrate or args.report
    do_run = args.run or args.report
    if not (do_calibrate or do_run):
        parser.error("pick --calibrate, --run or --report")

    manifest_path = Path(args.manifest)
    index_path = Path(args.index) if args.index else index_path_for(manifest_path)
    if not index_path.is_file():
        print(
            f"corpus index not found: {index_path}\n"
            "build it with: scripts/postsession_corpus.py --build",
            file=sys.stderr,
        )
        return 2
    manifest = read_manifest(manifest_path) if manifest_path.is_file() else None

    gate = Gate(DEFAULT_GATE)
    started = time.monotonic()
    splits_read: list[str] = []
    calibration = None
    mining_report = None
    results: list[SessionExtraction] = []
    judges: list[Any] = []
    deduper = build_deduper(Path(args.db) if args.db else None)

    if do_calibrate:
        splits_read.extend(ALLOWED_SPLITS)
        entries = index_entries(index_path, ALLOWED_SPLITS)
        if args.progress:
            print(f"calibrating over {len(entries)} sessions", file=sys.stderr)
        cohorts = collect_cohorts(entries, max_bytes=args.max_bytes, progress=args.progress)
        calibration = calibrate(cohorts, gate)

    if do_run:
        if "eval" not in splits_read:
            splits_read.append("eval")
        entries = index_entries(index_path, ("eval",))
        if args.max_sessions:
            entries = sorted(entries, key=lambda item: _sample_rank(item.session_key))
            entries = entries[: args.max_sessions]
        entries = [entry for entry in entries if entry.bytes <= args.max_bytes]
        config = ExtractionConfig(
            mining=DEFAULT_MINING, max_judge_calls=args.max_judge_calls
        )
        if args.progress:
            print(f"extracting over {len(entries)} eval sessions", file=sys.stderr)
        results, mining_report, judges = run_extraction(
            entries,
            judge_factory=lambda: build_judge(
                args.judge,
                model=args.model,
                cassette=Path(args.cassette) if args.cassette else None,
            ),
            deduper=deduper,
            gate=gate,
            config=config,
            judge_budget=args.judge_budget,
            workers=args.workers,
            progress=args.progress,
        )

    report = build_report(
        splits_read=sorted(dict.fromkeys(splits_read)),
        manifest=manifest,
        calibration=calibration,
        mining_report=mining_report,
        results=results,
        judges=judges,
        dedup_available=deduper.available,
        elapsed_s=time.monotonic() - started,
        max_ops=args.max_ops,
    )
    out = write_report(report, Path(args.out))
    print(f"wrote {out}")
    if calibration:
        for split in ALLOWED_SPLITS:
            block = calibration[split]
            print(
                f"  gate[{split}]: journal n={block['journal']['n']} "
                f"rule={block['journal']['journal_rule']:.3f} | "
                f"organic n={block['organic']['n']} "
                f"rule={block['organic']['journal_rule']:.3f} | "
                f"gap={block['discrimination']['journal_rule_gap']:+.3f}"
            )
    if mining_report:
        print(
            f"  mining: {mining_report['candidates']} candidates over "
            f"{mining_report['sessions_scanned']} sessions "
            f"({mining_report['sessions_with_candidates']} with >=1)"
        )
        print(f"  by kind: {json.dumps(mining_report['by_kind'], sort_keys=True)}")
    if results:
        summary = report["extraction"]
        print(
            f"  extraction: {summary['proposed']} proposed / "
            f"{summary['candidates']} candidates in {summary['sessions']} sessions"
        )
        print(f"  rejections: {json.dumps(summary['rejection_reasons'], sort_keys=True)}")
        print(f"  structural checks pass: {report['structural_checks']['all_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
