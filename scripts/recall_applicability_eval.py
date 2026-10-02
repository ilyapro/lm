#!/usr/bin/env python3
"""Paired, private recall/lookup measurement on one frozen SQLite snapshot.

Case JSON: {"cases": [{"case_id", "query", "scope", "depth", "max_results",
"category", "required_facts": [{"text", "acceptable_source_ids": [...] }]}]}.
``acceptable_source_ids`` may instead be a case-level list applying to all facts.
An absent-knowledge case uses an empty required_facts list. IDs are oracle-only:
the worker receives only query, scope, depth and max_results. Reports are local
and include source IDs; keep both input and output outside tracked artifacts.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "b9769d8"


def wire(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cases(raw: Any) -> list[dict[str, Any]]:
    cases = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(cases, list) or not cases:
        raise ValueError("case file needs a nonempty cases list")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or any(key not in case for key in
            ("case_id", "query", "scope", "depth", "max_results", "category", "required_facts")):
            raise ValueError("case missing a required field")
        if not isinstance(case["case_id"], str) or not case["case_id"] or case["case_id"] in seen:
            raise ValueError("invalid or repeated case_id")
        seen.add(case["case_id"])
        if not isinstance(case["query"], str) or not case["query"].strip():
            raise ValueError(f"{case['case_id']}: empty query")
        if not isinstance(case["scope"], (str, type(None))) or not isinstance(case["max_results"], int) or case["max_results"] < 1:
            raise ValueError(f"{case['case_id']}: invalid recall parameters")
        if not isinstance(case["category"], str) or not isinstance(case["required_facts"], list):
            raise ValueError(f"{case['case_id']}: invalid category or facts")
        for fact in case["required_facts"]:
            if not isinstance(fact, dict) or not isinstance(fact.get("text"), str) or not fact["text"]:
                raise ValueError(f"{case['case_id']}: facts need nonempty text")
            ids = fact.get("acceptable_source_ids", case.get("acceptable_source_ids"))
            if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids):
                raise ValueError(f"{case['case_id']}: every fact needs acceptable source IDs")
            fact["acceptable_source_ids"] = ids
    return cases


def _source_for(content: str, index: int, node_id: str) -> str:
    """Associate consolidated evidence text with its nearest preceding source line."""
    markers = list(re.finditer(r"(?m)^\s*-\s+([A-Za-z0-9_-]+):", content[:index]))
    return markers[-1].group(1) if markers else node_id


def _matches(content: str, node_id: str, fact: dict[str, Any]) -> bool:
    ids = set(fact["acceptable_source_ids"])
    start = 0
    while (index := content.find(fact["text"], start)) >= 0:
        if _source_for(content, index, node_id) in ids:
            return True
        start = index + 1
    return False


def _entries(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [item.get("node", item) for item in response.get("results", [])
            if isinstance(item, dict) and isinstance(item.get("node", item), dict)]


def score(case: dict[str, Any], transcript: dict[str, Any], source_content: dict[str, str]) -> dict[str, Any]:
    calls = transcript["calls"]
    facts = case["required_facts"]
    first = calls[0]["response"]
    ranks = {node["id"]: rank for rank, node in enumerate(_entries(first), 1)}
    relevant = {node_id for node_id, content in source_content.items()
                if any(_matches(content, node_id, fact) for fact in facts)}
    reached: set[int] = set()
    first_sufficient: int | None = None
    for call_number, call in enumerate(calls, 1):
        for node in _entries(call["response"]):
            content = node.get("content")
            if isinstance(content, str):
                reached.update(i for i, fact in enumerate(facts) if _matches(content, node["id"], fact))
        if first_sufficient is None and facts and len(reached) == len(facts):
            first_sufficient = call_number
    return {
        "case_id": case["case_id"], "category": case["category"],
        "required_fact_count": len(facts), "reached_fact_indexes": sorted(reached),
        "sufficient": first_sufficient is not None if facts else not bool(first.get("results")),
        "necessary_call_count": first_sufficient,
        "necessary_sequence": [call["tool"] for call in calls[:first_sufficient]] if first_sufficient else None,
        "relevant_ranks": sorted(ranks[x] for x in relevant if x in ranks),
        "irrelevant_ranks": sorted(rank for node_id, rank in ranks.items() if node_id not in relevant),
        "irrelevant_top_four": sum(node_id not in relevant for node_id, rank in ranks.items() if rank <= 4),
        "call_count": len(calls), "recall_count": sum(x["tool"] == "memory_recall" for x in calls),
        "lookup_count": sum(x["tool"] == "memory_lookup" for x in calls),
        "response_bytes": sum(x["response_bytes"] for x in calls),
        "necessary_response_bytes": sum(x["response_bytes"] for x in calls[:first_sufficient]) if first_sufficient else None,
        "warm_latency_ms": round(sum(x["latency_ms"] for x in calls), 3),
        "calls": [{"tool": x["tool"], "response_bytes": x["response_bytes"],
                   "latency_ms": x["latency_ms"]} for x in calls],
    }


def source_contents(snapshot: Path) -> dict[str, str]:
    con = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
    try:
        return {str(row[0]): str(row[1]) for row in con.execute("SELECT id, content FROM nodes")}
    finally:
        con.close()


def _archive_baseline(destination: Path) -> None:
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", BASELINE, "src"],
                             check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(destination, filter="data")


def production_delta(baseline_root: Path, candidate_root: Path) -> dict[str, Any]:
    """Simple reviewable source delta, independent of the runner's own lines."""
    files = ("src/living_memory/retrieval.py", "src/living_memory/score_gate.py")
    changes: dict[str, Any] = {}
    for name in files:
        before = (baseline_root / name).read_text().splitlines()
        after = (candidate_root / name).read_text().splitlines()
        added = removed = 0
        for tag, a, b, c, d in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
            if tag != "equal":
                removed += b - a
                added += d - c
        changes[name] = {"baseline_lines": len(before), "candidate_lines": len(after),
                         "added_lines": added, "removed_lines": removed}
    return changes


def run_pair(snapshot: Path, cases: list[dict[str, Any]], baseline_root: Path | None,
             candidate_root: Path, timeout: int = 300) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    if not snapshot.is_file():
        raise ValueError(f"snapshot missing: {snapshot}")
    snapshot_hash = sha256(snapshot)
    content = source_contents(snapshot)
    with tempfile.TemporaryDirectory(prefix="lm-applicability-") as temp:
        work = Path(temp)
        if baseline_root is None:
            baseline_root = work / "baseline-code"
            baseline_root.mkdir()
            _archive_baseline(baseline_root)
        arms = {"baseline": baseline_root.resolve(), "candidate": candidate_root.resolve()}
        delta = production_delta(arms["baseline"], arms["candidate"])
        results: dict[str, dict[str, Any]] = {}
        for index, case in enumerate(cases):
            measured: dict[str, Any] = {}
            order = list(arms) if index % 2 == 0 else list(reversed(arms))
            for arm in order:
                db = work / f"{index}-{arm}.sqlite3"
                src = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
                dst = sqlite3.connect(db)
                try:
                    src.backup(dst)
                finally:
                    dst.close(); src.close()
                payload = {key: case[key] for key in ("query", "scope", "depth", "max_results")}
                env = os.environ.copy()
                env["PYTHONPATH"] = str(arms[arm] / "src")
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(db)],
                    input=wire(payload), capture_output=True, env=env, timeout=timeout)
                if proc.returncode:
                    raise RuntimeError(f"{case['case_id']} {arm} worker failed: {proc.stderr.decode(errors='replace')}")
                transcript = json.loads(proc.stdout)
                measured[arm] = score(case, transcript, content)
            base, candidate = measured["baseline"], measured["candidate"]
            candidate["lost_fact_indexes"] = sorted(set(base["reached_fact_indexes"]) - set(candidate["reached_fact_indexes"]))
            results[case["case_id"]] = measured
    return {"snapshot_sha256": snapshot_hash, "baseline_commit": BASELINE,
            "arm_order": "alternating by case index", "environment": "inherited identically; PYTHONPATH differs only by code root",
            "warmup": "create_mcp_server warms LocalEmbeddingModel before timed calls; fresh process and DB per case and arm; first-query vector/index work remains timed",
            "production_code_delta": delta,
            "cases": results}


def worker(db: Path) -> None:
    class LocalMCP:
        def __init__(self, name: str, **kwargs: Any):
            self.tools: dict[str, Any] = {}
        def tool(self, func: Any = None, **kwargs: Any) -> Any:
            def register(f: Any) -> Any:
                self.tools[str(kwargs.get("name") or f.__name__)] = f
                return f
            return register(func) if func else register
        def resource(self, uri: str, **kwargs: Any) -> Any:
            return lambda func: func
        def prompt(self, **kwargs: Any) -> Any:
            return lambda func: func

    from living_memory.server import create_mcp_server
    case = json.loads(sys.stdin.buffer.read())
    mcp = create_mcp_server(db, mcp_factory=LocalMCP)
    calls: list[dict[str, Any]] = []
    def call(name: str, **kwargs: Any) -> dict[str, Any]:
        start = time.perf_counter_ns()
        response = mcp.tools[name](**kwargs)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        calls.append({"tool": name, "response": response,
                      "response_bytes": len(wire(response)), "latency_ms": round(elapsed, 3)})
        return response
    response = call("memory_recall", query=case["query"], scope=case["scope"],
                    depth=case["depth"], max_results=case["max_results"],
                    ambient_context={"transport_session_id": "paired-fresh-session"})
    # Follow only IDs supplied by recall; the worker never receives gold IDs.
    for item in response.get("results", []):
        ref = item.get("content_ref")
        if isinstance(ref, dict) and isinstance(ref.get("node_id"), str):
            call("memory_lookup", node_id=ref["node_id"])
    sys.stdout.buffer.write(wire({"calls": calls}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not all((args.snapshot, args.cases, args.out)):
        parser.error("--snapshot, --cases, and --out are required")
    cases = validate_cases(json.loads(args.cases.read_text()))
    report = run_pair(args.snapshot, cases, args.baseline_root, args.candidate_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(wire(report) + b"\n")
    print(f"{len(cases)} paired cases -> {args.out}")


if __name__ == "__main__":
    main()
