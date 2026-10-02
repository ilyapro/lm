#!/usr/bin/env python3
"""Validate a private applicability corpus and capture pre-candidate baseline ranks.

The private JSON and SQLite database must stay outside git. This script prints
only counts and hashes; baseline output is written beside the private corpus.
The later paired runner is the authority for delivery and lookup metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate(snapshot: Path, corpus: Path) -> dict:
    packet = json.loads(corpus.read_text(encoding="utf-8"))
    cases = packet["cases"]
    if packet.get("schema_version") != 1 or len(cases) < 16:
        raise ValueError("invalid schema or case count")
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id")
    holdout = [case for case in cases if case["split"] == "holdout"]
    development = [case for case in cases if case["split"] == "development"]
    if len(holdout) < 12 or not development:
        raise ValueError("need development and at least twelve holdout cases")
    if len({case["query"] for case in cases}) != len(cases):
        raise ValueError("query reused across splits")
    groups = {split: {case["topic_group"] for case in cases if case["split"] == split}
              for split in ("development", "holdout")}
    if groups["development"] & groups["holdout"]:
        raise ValueError("development/holdout topic groups overlap")
    sources = {split: {source for case in cases if case["split"] == split
                       for source in case["acceptable_source_ids"]}
               for split in ("development", "holdout")}
    if sources["development"] & sources["holdout"]:
        raise ValueError("development/holdout answer sources overlap")
    if len(sources["holdout"]) != sum(len(c["acceptable_source_ids"]) for c in holdout):
        raise ValueError("holdout answer source reused")
    categories = Counter(case["category"] for case in holdout)
    required = {"concrete", "compound", "instruction", "correction",
                "cross-project", "absent-knowledge", "large-group"}
    if not required <= categories.keys():
        raise ValueError("missing holdout category")
    db = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
    try:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("snapshot failed quick_check")
        for case in cases:
            if case["max_results"] != 4 or case["depth"] != 1:
                raise ValueError("recall parameters differ")
            if not case["query"].strip() or not case["topic_group"]:
                raise ValueError("empty query/group")
            if case["category"] == "absent-knowledge":
                if case["required_facts"] or case["acceptable_source_ids"]:
                    raise ValueError("absent case has answer key")
            elif not case["required_facts"] or not case["acceptable_source_ids"]:
                raise ValueError("positive case lacks oracle")
            for source in case["acceptable_source_ids"]:
                row = db.execute("SELECT content FROM nodes WHERE id=?", (source,)).fetchone()
                if row is None or any(fact not in row[0] for fact in case["required_facts"]):
                    raise ValueError(f"oracle source missing facts in {case['case_id']}")
    finally:
        db.close()
    return {"snapshot_sha256": digest(snapshot), "corpus_sha256": digest(corpus),
            "case_count": len(cases), "development_count": len(development),
            "holdout_count": len(holdout), "holdout_categories": dict(sorted(categories.items()))}


def baseline(snapshot: Path, corpus: Path, output: Path) -> None:
    manifest = validate(snapshot, corpus)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from living_memory.config import MemoryConfig
    from living_memory.retrieval import MemoryRecallService
    from living_memory.storage import MemoryStore

    cases = json.loads(corpus.read_text(encoding="utf-8"))["cases"]
    rows = []
    with tempfile.TemporaryDirectory(prefix="recall-applicability-") as tmp:
        work = Path(tmp) / "baseline.sqlite3"
        shutil.copyfile(snapshot, work)
        store = MemoryStore(MemoryConfig(db_path=work))
        try:
            for case in cases:
                # Fresh service removes service-local history; event/access logging
                # is disabled so case order does not write feedback into the copy.
                service = MemoryRecallService(store)
                started = time.perf_counter()
                found = service.memory_recall(
                    case["query"], scope=case["scope"], depth=case["depth"],
                    max_results=case["max_results"], log_access=False, log_event=False,
                )
                ranked = [result.node_id for result in found]
                source_ranks = {source: ranked.index(source) + 1
                                for source in case["acceptable_source_ids"] if source in ranked}
                rows.append({"case_id": case["case_id"], "split": case["split"],
                             "ranked_ids": ranked, "source_ranks": source_ranks,
                             "recall_seconds": time.perf_counter() - started})
        finally:
            store.close()
    output.write_text(json.dumps({"schema_version": 1, "recorded_at": datetime.now(UTC).isoformat(),
                                  "code_head": "b9769d84e3188ee1e646627ebe1b4d454f69d6f9",
                                  "inputs": manifest, "outcomes": rows},
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "baseline_sha256": digest(output)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "baseline"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(validate(args.snapshot, args.corpus), sort_keys=True))
    elif args.out is None:
        parser.error("baseline requires --out")
    else:
        baseline(args.snapshot, args.corpus, args.out)


if __name__ == "__main__":
    main()
