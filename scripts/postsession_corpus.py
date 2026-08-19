#!/usr/bin/env python3
"""Build, verify and describe the sealed post-session session corpus.

    scripts/postsession_corpus.py --build     # enumerate, hash, split, seal
    scripts/postsession_corpus.py --verify    # check the seal against disk
    scripts/postsession_corpus.py --stats     # describe the corpus

Two files come out of ``--build`` and they are tracked differently:

``artifacts/post-session/corpus.json``
    the manifest -- roots, counts, split policy, per-split seal digests,
    ``holdout_sha256`` and ``index_sha256``. Tracked, ~130 lines, no rows.

``artifacts/post-session/corpus-index.jsonl``
    one compact JSON object per session. Thousands of absolute paths under the
    user's home directory, fully rebuildable from the roots, so ``.gitignore``
    keeps it out of the repository.

This script is READ-ONLY with respect to everything it scans. It never writes
outside ``--manifest`` / ``--index`` (and ``--report`` / ``--stats-out`` when
asked), never touches the Living Memory database, and never modifies anything
under the AE project tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from living_memory.postsession.corpus import (  # noqa: E402
    DEFAULT_MANIFEST,
    SPLITS,
    build_corpus,
    index_path_for,
    iter_index,
    load_session,
    read_manifest,
    recovery_stats,
    roots_from_manifest,
    verify_corpus,
    write_corpus,
)
from living_memory.postsession.transcripts import ADAPTERS, Roots  # noqa: E402


def _display(path: Path) -> str:
    """Repo-relative when it is inside the repo, so the manifest travels."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _paths(args: argparse.Namespace) -> tuple[Path, Path]:
    manifest = Path(args.manifest)
    index = Path(args.index) if args.index else index_path_for(manifest)
    return manifest, index


def _build(args: argparse.Namespace) -> int:
    roots = Roots.discover(home=args.home, ae_root=args.ae_root)
    manifest_path, index_path = _paths(args)
    started = time.monotonic()

    def progress(seen: int, _ref: Any) -> None:
        if args.progress and seen % 250 == 0:
            print(f"  ... {seen} transcripts hashed", file=sys.stderr, flush=True)

    corpus = build_corpus(
        roots,
        sources=args.source or None,
        limit=args.limit,
        index_display_path=_display(index_path),
        progress=progress,
    )
    write_corpus(corpus, manifest_path, index_path)
    counts = corpus.manifest["counts"]
    print(f"wrote {manifest_path} in {time.monotonic() - started:.1f}s")
    print(f"  index -> {index_path} ({corpus.manifest['index']['lines']} rows)")
    print(
        "  sessions={sessions} bytes={bytes} groups={identity_groups} "
        "largest_group={largest_group}".format(**counts)
    )
    print("  by_split=" + json.dumps(counts["by_split"], sort_keys=True))
    print("  by_source=" + json.dumps(counts["by_source"], sort_keys=True))
    print("  holdout_sha256=" + corpus.manifest["holdout_sha256"])
    print("  index_sha256=" + corpus.manifest["index_sha256"])
    return 0


def _verify(args: argparse.Namespace) -> int:
    manifest_path, index_path = _paths(args)
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = read_manifest(manifest_path)
    roots = (
        Roots.discover(home=args.home, ae_root=args.ae_root)
        if (args.home or args.ae_root)
        else roots_from_manifest(manifest)
    )
    report = verify_corpus(
        manifest, index_path=index_path, roots=roots, strict=args.strict
    )

    if report["mode"] == "index":
        index = report["index"]
        print(
            f"index {index['path']}: "
            f"{'ok' if index['sha256_match'] else 'DIGEST MISMATCH'} "
            f"({index['actual_lines']} rows)"
        )
        print(
            f"verified {report['verified']}/{report['checked']} transcripts "
            "byte-identical"
        )
        cross = report["cross_check"]
        print(f"  cross-check counts: {'ok' if cross['counts_match'] else 'MISMATCH'}")
        for name in SPLITS:
            seal = cross["seal"][name]
            state = "ok" if seal["match"] else "MISMATCH"
            print(f"  seal[{name}]: {state} {seal['recomputed']}")
        print(f"  split re-derivation mismatches: {len(report['split_mismatches'])}")
        appended = sum(1 for item in report["changed"] if item["kind"] == "append")
        print(
            f"  changed: {len(report['changed'])} (append {appended} / "
            f"rewrite {len(report['changed']) - appended})  "
            f"missing: {len(report['missing'])}"
        )
        print(
            f"  holdout appended: {len(report['holdout_appended'])}  "
            f"rewritten: {len(report['holdout_rewritten'])}  "
            f"missing: {len(report['holdout_missing'])}"
        )
    else:
        rederived = report["rederived"]
        print(f"no index at {index_path}: re-deriving digests from disk")
        print(f"  sessions now on disk: {rederived['counts']['sessions']}")
        print(f"  by_split now: {json.dumps(rederived['counts']['by_split'], sort_keys=True)}")
        for name in SPLITS:
            seal = rederived["seal"][name]
            print(f"  seal[{name}]: {'same' if seal['match'] else 'DRIFTED'}")
        print(f"  count drift: {json.dumps(rederived['counts_drift'], sort_keys=True)}")
        print("  (rebuild with --build to re-seal; drift is reported, not failed)")

    if not report["content_ok"]:
        print(f"  CONTENT LEAK: {report.get('content_error')}", file=sys.stderr)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"  report -> {args.report}")
    print("OK" if report["ok"] else "FAILED")
    return 0 if report["ok"] else 1


def _stats(args: argparse.Namespace) -> int:
    manifest_path, index_path = _paths(args)
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = read_manifest(manifest_path)
    report: dict[str, Any] = {
        "manifest": str(manifest_path),
        "generated_at": manifest.get("generated_at"),
        "counts": manifest.get("counts"),
        "holdout_sha256": manifest.get("holdout_sha256"),
        "split_sha256": manifest.get("split_sha256"),
        "index_sha256": manifest.get("index_sha256"),
        "index_present": index_path.is_file(),
    }
    if index_path.is_file():
        entries = list(iter_index(index_path))
        report["time_range"] = {
            "earliest": min((e.started_at for e in entries if e.started_at), default=None),
            "latest": max((e.ended_at for e in entries if e.ended_at), default=None),
        }
        report["top_repos"] = _top(entries, lambda e: e.repo, 12)
        report["models"] = _top(entries, lambda e: e.model, 12)
        if args.deep:
            report["deep"] = _deep(entries, args)
    elif args.deep:
        report["deep"] = {"error": f"no index at {index_path}; run --build first"}
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    if args.stats_out:
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


def _top(entries: Any, key: Any, limit: int) -> dict[str, int]:
    counter: dict[str, int] = {}
    for entry in entries:
        value = key(entry)
        if value:
            counter[str(value)] = counter.get(str(value), 0) + 1
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:limit])


def _deep(entries: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Actually parse a deterministic sample and report what was recovered."""

    wanted = set(args.deep_split or ("train", "eval"))
    # --source restricts the deep sample too. Without this the flag was accepted
    # and ignored, so `--deep --source deepseek` and `--deep --source gigacode`
    # returned byte-identical reports describing neither.
    sources = set(args.source or ADAPTERS)
    pool = sorted(
        (
            entry
            for entry in entries
            if entry.split in wanted and entry.source in sources
        ),
        key=lambda item: item.sha256,
    )
    sample = pool[: args.deep_limit]
    records = []
    failures: list[dict[str, str]] = []
    for entry in sample:
        try:
            records.append(load_session(entry))
        except Exception as error:  # noqa: BLE001 - the point is to report them
            # --stats --deep is a survey: a transcript the normalizer cannot
            # parse is the finding, so it is named rather than allowed to abort
            # the run over the other few thousand.
            failures.append(
                {
                    "session_key": entry.session_key,
                    "path": entry.path,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    stats = recovery_stats(records)
    stats["sampled"] = len(sample)
    stats["parsed"] = len(records)
    stats["failures"] = failures
    stats["splits"] = sorted(wanted)
    stats["sources"] = sorted(sources)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true", help="rebuild manifest and index")
    mode.add_argument("--verify", action="store_true", help="check the seal against disk")
    mode.add_argument("--stats", action="store_true", help="describe the corpus")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--index", default=None, help="per-session index (default: manifest sibling)"
    )
    parser.add_argument("--home", default=None, help="override the home directory scanned")
    parser.add_argument("--ae-root", default=None, help="override the AE project root")
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(ADAPTERS),
        help="restrict to one source (repeatable)",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N transcripts")
    parser.add_argument("--progress", action="store_true", help="log hashing progress")
    parser.add_argument("--strict", action="store_true", help="verify: fail on any drift")
    parser.add_argument("--report", default=None, help="verify: write the JSON report here")
    parser.add_argument("--deep", action="store_true", help="stats: parse a sample")
    parser.add_argument("--deep-limit", type=int, default=50)
    parser.add_argument("--deep-split", action="append", choices=sorted(SPLITS))
    parser.add_argument("--stats-out", default=None)
    args = parser.parse_args(argv)

    if args.build:
        return _build(args)
    if args.verify:
        return _verify(args)
    return _stats(args)


if __name__ == "__main__":
    raise SystemExit(main())
