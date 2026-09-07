#!/usr/bin/env python3
"""Byte-identity fixture for ``living_memory.embeddings.tokenize`` on non-Cyrillic input.

Runs a *previous* tokenizer -- the module file dumped from git, e.g.
``git show master:src/living_memory/embeddings.py > /tmp/master_embeddings.py``
-- over a sample that contains no Cyrillic letters and writes each input with
the token list that tokenizer produced. ``tests/test_tokenize_cyrillic.py``
then pins that the checkout's tokenizer reproduces every record exactly, which
is the evidence that Cyrillic stemming changed nothing for Latin text and
identifiers.

Sample: every distinct goldset query without Cyrillic letters, plus a seeded
random sample of active node contents without Cyrillic letters from a
read-only database snapshot, each content cut at ``--content-chars``.

Usage::

    python3 scripts/tokenize_master_fixture.py \\
        --master-module /tmp/master_embeddings.py \\
        --goldset ~/.cache/living-memory-harness/recalib/goldset-recalibration.jsonl \\
        --snapshot ~/.cache/living-memory-harness/usage-signal/sfx-2026-09-07.sqlite3 \\
        --out tests/fixtures/tokenize_master_non_cyrillic.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import random
import re
import sqlite3
from pathlib import Path

CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("previous_embeddings", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--master-module", required=True, help="embeddings.py of the previous tokenizer")
    parser.add_argument("--goldset", required=True, help="goldset JSONL whose queries to include")
    parser.add_argument("--snapshot", required=True, help="read-only SQLite snapshot for node contents")
    parser.add_argument("--contents", type=int, default=300, help="node contents to sample")
    parser.add_argument("--content-chars", type=int, default=1000, help="cut each content here")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--out", required=True, help="gzipped JSONL fixture path")
    args = parser.parse_args(argv)

    previous = load_module(Path(args.master_module))
    queries: list[str] = []
    seen: set[str] = set()
    with Path(args.goldset).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            query = str(json.loads(line)["query"])
            if CYRILLIC_RE.search(query) or query in seen:
                continue
            seen.add(query)
            queries.append(query)

    connection = sqlite3.connect(f"file:{Path(args.snapshot).resolve()}?mode=ro", uri=True)
    try:
        rows = [
            str(row[0])
            for row in connection.execute("SELECT content FROM nodes WHERE decayed = 0 ORDER BY id")
        ]
    finally:
        connection.close()
    candidates = [row for row in rows if row and not CYRILLIC_RE.search(row)]
    random.seed(args.seed)
    contents = [text[: args.content_chars] for text in random.sample(candidates, args.contents)]

    records = [{"kind": "goldset_query", "text": query} for query in queries]
    records += [{"kind": "node_content", "text": text} for text in contents]
    for record in records:
        record["tokens"] = previous.tokenize(record["text"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=9) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"{len(queries)} queries + {len(contents)} contents "
        f"({sum(len(record['tokens']) for record in records)} tokens) -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
