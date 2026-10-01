#!/usr/bin/env python3
"""Frozen, paired recall cost replay. Private packet and full report stay local.

Packet v1/v2 follows ``artifacts/recall-needed-cost/prereg.md``: cases carry
``query``, ``scope``, ``max_results`` and an independent ``oracle`` with
``required_source_ids``, ``must_contain``, ``must_not_assert`` and
``absence_expected``. Literal clauses are compared after whitespace folding.
An optional ``queries`` list permits conditional later recalls. The older
internal ``required`` form is accepted for small synthetic probes.

Run from a clean, committed candidate checkout::

  python3 scripts/recall_needed_cost.py run --packet PRIVATE.json \\
    --packet-sha256 HASH --snapshot SNAP.sqlite3 --snapshot-sha256 HASH \\
    --baseline-ref BASELINE_COMMIT --candidate-ref HEAD --out PRIVATE_REPORT.json

The report contains ordered private node IDs, never response text or queries.
Keep it in local scratch. A separate aggregate-only JSON is written with
``--aggregate-out``. Both revisions run in fresh subprocesses with imports
verified under their own git archives, each case on a fresh byte-identical
copy of the verified SQLite snapshot. No live store is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path: Path, expected: str) -> None:
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"invalid SHA-256 for {path.name}")
    if sha256(path) != expected:
        raise ValueError(f"frozen hash mismatch: {path.name}")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def response_chars(value: Any) -> int:
    return len(compact_json(value))


def _field(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _contains(value: Any, needle: str) -> bool:
    folded = " ".join(needle.split())
    if isinstance(value, str):
        return folded in " ".join(value.split())
    if isinstance(value, dict):
        return any(_contains(item, needle) for item in value.values())
    if isinstance(value, list):
        return any(_contains(item, needle) for item in value)
    return folded in compact_json(value)


def requirement_met(entry: dict[str, Any], requirement: dict[str, Any]) -> bool:
    """Independent oracle over what a tool response actually delivered."""
    if entry.get("id") not in requirement["node_ids"]:
        return False
    field = requirement.get("field", "content")
    if not isinstance(field, str) or not field:
        raise ValueError("requirement field must be a nonempty path")
    needle = requirement.get("text")
    if not isinstance(needle, str) or not needle:
        raise ValueError("requirement text must be nonempty")
    return _contains(entry if field == "*" else _field(entry, field), needle)


def remaining(requirements: list[dict[str, Any]], observed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [need for need in requirements if not any(requirement_met(node, need) for node in observed)]


def validate_packet(packet: Any) -> list[dict[str, Any]]:
    if not isinstance(packet, dict) or packet.get("schema_version", packet.get("version")) not in (1, 2):
        raise ValueError("packet version must be 1 or 2")
    cases = packet.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("packet needs nonempty cases")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
            raise ValueError("case id required")
        if case["id"] in seen:
            raise ValueError("duplicate case id")
        seen.add(case["id"])
        if case.get("split") not in ("development", "holdout"):
            raise ValueError("case split must be development or holdout")
        if "oracle" in case:
            oracle = case["oracle"]
            if not isinstance(oracle, dict):
                raise ValueError("oracle must be an object")
            ids = oracle.get("required_source_ids")
            clauses = oracle.get("must_contain")
            forbidden = oracle.get("must_not_assert")
            absent = oracle.get("absence_expected")
            if not isinstance(ids, list) or not isinstance(clauses, list) or not isinstance(forbidden, list) or not isinstance(absent, bool):
                raise ValueError("invalid oracle fields")
            if not all(isinstance(x, str) and x for x in ids + clauses + forbidden):
                raise ValueError("oracle IDs and clauses must be nonempty strings")
            if absent and (ids or clauses):
                raise ValueError("missing control cannot claim required sources")
            if not absent and (not ids or not clauses):
                raise ValueError("positive case needs source IDs and clauses")
            required = [{"node_ids": ids, "text": text, "field": "*"} for text in clauses]
            expected = "missing" if absent else "sufficient"
            queries = case.get("queries") or [{
                "query": case.get("query"), "scope": case.get("scope"),
                "max_results": case.get("max_results", 5),
            }]
        else:
            expected = case.get("expected")
            required = []
            for need in case.get("required", []):
                if not isinstance(need, dict):
                    raise ValueError("requirement must be an object")
                required.append({**need, "node_ids": need.get("node_ids", [need.get("node_id")])})
            ids = list(dict.fromkeys(node_id for need in required for node_id in need["node_ids"]))
            forbidden = []
            queries = case.get("queries")
        if expected not in ("sufficient", "missing") or not isinstance(queries, list) or not queries:
            raise ValueError("case needs expected outcome and queries")
        if expected == "sufficient" and not required:
            raise ValueError("positive case needs required text")
        for query in queries:
            if not isinstance(query, dict) or not isinstance(query.get("query"), str) or not query["query"]:
                raise ValueError("query must be an object with nonempty query")
        for need in required:
            if not isinstance(need.get("node_ids"), list) or not all(isinstance(x, str) and x for x in need["node_ids"]):
                raise ValueError("requirement needs source IDs")
            if not isinstance(need.get("text"), str) or not need["text"]:
                raise ValueError("requirement needs nonempty text")
        normalized.append({"id": case["id"], "split": case["split"], "expected": expected,
                           "queries": queries, "required": required, "source_ids": ids,
                           "forbidden": forbidden})
    return normalized


class FakeMCP:
    """Register real tool functions without a network transport."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.instructions = kwargs.get("instructions")
        self.tools: dict[str, Any] = {}

    def tool(self, func: Any = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner
        return decorate(func) if func is not None else decorate

    def resource(self, *args: Any, **kwargs: Any) -> Any:
        return lambda inner: inner

    def prompt(self, func: Any = None, **kwargs: Any) -> Any:
        return (lambda inner: inner) if func is None else func


def _worker(args: argparse.Namespace) -> None:
    source = Path(args.source_root).resolve()
    sys.path.insert(0, str(source / "src"))
    import living_memory  # noqa: PLC0415
    from living_memory.server import create_mcp_server  # noqa: PLC0415

    loaded = Path(living_memory.__file__).resolve()
    if not loaded.is_relative_to(source / "src"):
        raise RuntimeError(f"revision import escaped archive: {loaded}")
    packet = json.loads(Path(args.packet).read_text(encoding="utf-8"))
    cases = validate_packet(packet)
    reports: list[dict[str, Any]] = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="recall-case-") as directory:
            copy = Path(directory) / "store.sqlite3"
            shutil.copyfile(args.snapshot, copy)
            checked_file(copy, args.snapshot_sha256)
            server = create_mcp_server(copy, mcp_factory=FakeMCP)
            recall = server.tools["memory_recall"]
            lookup = server.tools["memory_lookup"]
            observed: list[dict[str, Any]] = []
            queried_ids: set[str] = set()
            responses: list[dict[str, Any]] = []
            ranking: list[list[str]] = []
            recall_ms = lookup_ms = 0.0
            count_lookup = 0
            for query in case["queries"]:
                start = time.perf_counter()
                result = recall(**query)
                recall_ms += (time.perf_counter() - start) * 1000
                ids = [str(item["node"]["id"]) for item in result["results"]]
                ranking.append(ids)
                responses.append({"tool": "memory_recall", "chars": response_chars(result)})
                observed.extend(item["node"] for item in result["results"])
                missing = remaining(case["required"], observed)
                # Only sources exposed by this recall may be fetched. Fetch all
                # currently necessary IDs in one lookup, then re-evaluate.
                to_fetch = list(dict.fromkeys(
                    source_id for need in missing for source_id in need["node_ids"]
                    if source_id in ids and source_id not in queried_ids
                ))
                if to_fetch:
                    start = time.perf_counter()
                    found = lookup(node_ids=to_fetch)
                    lookup_ms += (time.perf_counter() - start) * 1000
                    count_lookup += 1
                    responses.append({"tool": "memory_lookup", "chars": response_chars(found)})
                    queried_ids.update(to_fetch)
                    observed.extend(found["results"])
                if case["expected"] == "sufficient" and not remaining(case["required"], observed) and set(case["source_ids"]) <= {str(node.get("id")) for node in observed}:
                    break
            missing = remaining(case["required"], observed)
            source_missing = set(case["source_ids"]) - {str(node.get("id")) for node in observed}
            # Obsolete text is an assertion only when no delivered correction
            # supplies the complete current claim. Historical quoted evidence
            # alongside that correction remains permissible.
            old_unqualified = bool(case["forbidden"]) and bool(missing) and any(
                _contains(node.get("content"), phrase)
                for node in observed for phrase in case["forbidden"]
            )
            sufficient = case["expected"] == "sufficient" and not missing and not source_missing
            oracle_pass = (sufficient if case["expected"] == "sufficient" else True)
            if old_unqualified:
                oracle_pass = False
            reports.append({
                "id": case["id"], "split": case["split"],
                "expected": case["expected"], "sufficient": sufficient,
                "oracle_pass": oracle_pass,
                "remaining_requirements": len(missing) + len(source_missing),
                "ordered_retrieval_ids": ranking,
                "responses": responses,
                "recall_calls": len(ranking), "lookup_calls": count_lookup,
                "calls": len(responses), "response_chars": sum(x["chars"] for x in responses),
                "recall_ms": round(recall_ms, 3), "lookup_ms": round(lookup_ms, 3),
                "latency_ms": round(recall_ms + lookup_ms, 3),
            })
    Path(args.out).write_text(compact_json({"import_path": str(loaded), "cases": reports}) + "\n", encoding="utf-8")


def _revision(ref: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--verify", f"{ref}^{{commit}}"], text=True).strip()


def _archive(ref: str, destination: Path) -> None:
    archive = destination.parent / (destination.name + ".tar")
    with archive.open("wb") as stream:
        subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", ref], stdout=stream, check=True)
    destination.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")
    archive.unlink()


def _run(args: argparse.Namespace) -> None:
    packet_path = Path(args.packet).resolve()
    snapshot = Path(args.snapshot).resolve()
    checked_file(packet_path, args.packet_sha256)
    checked_file(Path(args.fixtures), args.fixtures_sha256)
    checked_file(snapshot, args.snapshot_sha256)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    validate_packet(packet)
    if packet.get("snapshot_sha256") != args.snapshot_sha256:
        raise ValueError("packet snapshot hash differs from frozen snapshot")
    revisions = {"baseline": _revision(args.baseline_ref), "candidate": _revision(args.candidate_ref)}
    if revisions["baseline"] == revisions["candidate"]:
        raise ValueError("baseline and candidate revisions must differ")
    with tempfile.TemporaryDirectory(prefix="recall-paired-") as directory:
        temp = Path(directory)
        arms: dict[str, Any] = {}
        for name, revision in revisions.items():
            source = temp / name
            _archive(revision, source)
            worker_out = temp / f"{name}.json"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(source / "src")
            subprocess.run([
                sys.executable, str(Path(__file__).resolve()), "_worker",
                "--source-root", str(source), "--snapshot", str(snapshot),
                "--snapshot-sha256", args.snapshot_sha256,
                "--packet", str(packet_path), "--out", str(worker_out),
            ], cwd=source, env=environment, check=True, timeout=args.timeout)
            arms[name] = json.loads(worker_out.read_text(encoding="utf-8"))["cases"]
        baseline, candidate = arms["baseline"], arms["candidate"]
        if [case["id"] for case in baseline] != [case["id"] for case in candidate]:
            raise RuntimeError("paired case order differs")
        ranking_changes = sum(a["ordered_retrieval_ids"] != b["ordered_retrieval_ids"] for a, b in zip(baseline, candidate))
        report = {
            "packet_sha256": args.packet_sha256, "fixtures_sha256": args.fixtures_sha256, "snapshot_sha256": args.snapshot_sha256,
            "revisions": revisions, "ranking_changed_cases": ranking_changes,
            "cases": [
                {"id": a["id"], "split": a["split"], "baseline": a, "candidate": b,
                 "ranking_changed": a["ordered_retrieval_ids"] != b["ordered_retrieval_ids"]}
                for a, b in zip(baseline, candidate)
            ],
        }
        aggregate: dict[str, Any] = {
            "packet_sha256": args.packet_sha256, "fixtures_sha256": args.fixtures_sha256, "snapshot_sha256": args.snapshot_sha256,
            "revisions": revisions, "cases": len(baseline),
            "ranking_changed_cases": ranking_changes, "by_split": {},
        }
        for split in ("development", "holdout"):
            left = [case for case in baseline if case["split"] == split]
            right = [case for case in candidate if case["split"] == split]
            aggregate["by_split"][split] = {
                name: {key: sum(case[key] for case in rows) for key in
                       ("response_chars", "calls", "recall_calls", "lookup_calls", "latency_ms", "oracle_pass")}
                for name, rows in (("baseline", left), ("candidate", right))
            }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.aggregate_out:
            aggregate_path = Path(args.aggregate_out)
            aggregate_path.parent.mkdir(parents=True, exist_ok=True)
            aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_FROZEN_HASHES = {
    "private_packet_sha256": "a54df4594910edb62d421fe7eec1add260022a6779809cdf710eef5619fd5d64",
    "snapshot_sha256": "2891a098b23c7ad0ef2a1c64a8bdfdd9dadccdccf4e2693afd31b09e55f09e69",
    "synthetic_fixtures_sha256": "972bd2e0e33baba2d2a03d2c0238deb46eae92bf303582943bbaf042f285b293",
}
_FROZEN_FAMILIES = (
    ("development", "known_large_carrier"),
    ("development", "ordinary_short_fact"),
    ("development", "ordinary_short_fact"),
    ("development", "buried_applicable_instruction"),
    ("development", "obsolete_rule_with_correction"),
    ("development", "missing_knowledge"),
    ("holdout", "ordinary_short_fact"),
    ("holdout", "buried_applicable_instruction"),
    ("holdout", "missing_knowledge"),
)
_TOTAL_FIELDS = ("response_chars", "calls", "recall_calls", "lookup_calls", "latency_ms", "oracle_pass")


def _verify_report(report: Any) -> None:
    """Validate the published aggregate without opening the private packet or replaying tools."""
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def mapping(value: Any, name: str) -> dict[str, Any]:
        require(isinstance(value, dict), f"{name} must be an object")
        return value

    def integer(value: Any, name: str, minimum: int = 0) -> int:
        require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
        return value

    def number(value: Any, name: str) -> float:
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                f"{name} must be a finite nonnegative number")
        return float(value)

    def same(actual: Any, expected: Any, name: str) -> None:
        require(type(actual) is type(expected) and actual == expected, f"{name} is inconsistent")

    def totals_match(actual: Any, expected: dict[str, Any], name: str) -> None:
        actual = mapping(actual, name)
        for key in _TOTAL_FIELDS:
            require(key in actual, f"{name}.{key} is missing")
            if key == "latency_ms":
                require(abs(number(actual[key], f"{name}.{key}") - expected[key]) <= 0.01,
                        f"{name}.{key} is inconsistent")
            else:
                same(actual[key], expected[key], f"{name}.{key}")

    report = mapping(report, "report")
    same(report.get("schema_version"), 1, "schema_version")
    require(isinstance(report.get("method"), str) and report["method"].strip(),
            "report method evidence missing")
    require(isinstance(report.get("limitations"), list) and
            all(isinstance(item, str) and item.strip() for item in report["limitations"]),
            "report limitations must be a list of statements")
    inputs = mapping(report.get("inputs"), "inputs")
    for key, expected in _FROZEN_HASHES.items():
        same(inputs.get(key), expected, f"inputs.{key}")
    revisions = mapping(report.get("revisions"), "revisions")
    baseline = revisions.get("baseline")
    candidate = revisions.get("candidate")
    same(baseline, "cd4ad6c3984c41badc55199299532199ed4579b1", "baseline revision")
    require(isinstance(candidate, str) and len(candidate) == 40 and
            all(char in "0123456789abcdef" for char in candidate) and candidate != baseline,
            "candidate revision must be a distinct full commit hash")
    try:
        actual_base = subprocess.check_output(
            ["git", "-C", str(ROOT), "merge-base", baseline, candidate],
            text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError("report revisions must be available commits") from error
    same(actual_base, baseline, "candidate merge-base provenance")

    complexity = mapping(report.get("production_complexity"), "production_complexity")
    same(complexity.get("merge_base"), baseline, "production_complexity.merge_base")
    files = complexity.get("files")
    require(isinstance(files, list) and set(files) == {
        "src/living_memory/delivery.py", "src/living_memory/server.py"},
        "production complexity needs both production files")
    for field in ("lines", "functions", "ast_branch_nodes"):
        left = integer(mapping(complexity.get("baseline"), "complexity baseline").get(field), field, 1)
        right = integer(mapping(complexity.get("candidate"), "complexity candidate").get(field), field, 1)
        delta = mapping(complexity.get("delta"), "complexity delta").get(field)
        same(delta, right - left, f"production_complexity.delta.{field}")
    diff = mapping(complexity.get("git_diff"), "production_complexity.git_diff")
    for field in ("insertions", "deletions"):
        integer(diff.get(field), f"production_complexity.git_diff.{field}")

    cases = report.get("cases")
    require(isinstance(cases, list) and len(cases) == len(_FROZEN_FAMILIES),
            "report needs all nine frozen cases")
    empty = lambda: {key: 0 for key in _TOTAL_FIELDS}
    totals = {arm: empty() for arm in ("baseline", "candidate")}
    splits = {split: {arm: empty() for arm in totals} for split in ("development", "holdout")}
    ranking_changes = savings_cases = 0
    for index, case in enumerate(cases):
        case = mapping(case, f"case {index + 1}")
        same(case.get("number"), index + 1, "case number")
        split, family = _FROZEN_FAMILIES[index]
        same(case.get("split"), split, f"case {index + 1} split")
        same(case.get("family"), family, f"case {index + 1} family")
        rows = {}
        for arm in totals:
            row = mapping(case.get(arm), f"case {index + 1} {arm}")
            rows[arm] = row
            expected = "missing" if family == "missing_knowledge" else "sufficient"
            same(row.get("expected"), expected, f"case {index + 1} {arm} expected")
            same(row.get("sufficient"), expected == "sufficient", f"case {index + 1} {arm} sufficient")
            same(row.get("oracle_pass"), True, f"case {index + 1} {arm} oracle_pass")
            same(row.get("remaining_requirements"), 0, f"case {index + 1} {arm} remaining_requirements")
            responses = row.get("responses")
            require(isinstance(responses, list) and responses, f"case {index + 1} {arm} needs responses")
            recall_count = lookup_count = chars = 0
            for position, response in enumerate(responses):
                response = mapping(response, "response")
                tool = response.get("tool")
                require(tool in ("memory_recall", "memory_lookup"), "invalid response tool")
                require(position > 0 or tool == "memory_recall", "first response must be recall")
                recall_count += tool == "memory_recall"
                lookup_count += tool == "memory_lookup"
                chars += integer(response.get("chars"), "response chars", 1)
            computed = {"response_chars": chars, "calls": len(responses),
                        "recall_calls": recall_count, "lookup_calls": lookup_count,
                        "oracle_pass": True}
            for key, value in computed.items():
                same(row.get(key), value, f"case {index + 1} {arm} {key}")
            recall_ms = number(row.get("recall_ms"), "recall_ms")
            lookup_ms = number(row.get("lookup_ms"), "lookup_ms")
            latency = number(row.get("latency_ms"), "latency_ms")
            require(recall_ms > 0 and (lookup_ms > 0) == (lookup_count > 0),
                    "latency evidence must cover each recorded tool type")
            require(abs(latency - recall_ms - lookup_ms) <= 0.01, "latency components disagree")
            computed["latency_ms"] = latency
            for key in _TOTAL_FIELDS:
                totals[arm][key] += computed[key]
                splits[split][arm][key] += computed[key]
        left, right = rows["baseline"], rows["candidate"]
        delta_chars = right["response_chars"] - left["response_chars"]
        delta_calls = right["calls"] - left["calls"]
        same(case.get("delta_chars"), delta_chars, f"case {index + 1} delta_chars")
        same(case.get("delta_calls"), delta_calls, f"case {index + 1} delta_calls")
        same(case.get("ranking_identical"), True, f"case {index + 1} ranking identity")
        if not case["ranking_identical"]:
            ranking_changes += 1
        savings_cases += delta_chars < 0
        if family != "missing_knowledge":
            require(delta_calls <= 0 and right["lookup_calls"] <= left["lookup_calls"],
                    f"case {index + 1} adds a mandatory call or lookup")
            require(not (right["lookup_calls"] > left["lookup_calls"] and delta_chars > 0),
                    f"case {index + 1} forced lookup cost regression")
    same(report.get("ranking_changed_cases"), ranking_changes, "ranking_changed_cases")
    published_totals = mapping(report.get("totals"), "totals")
    published_splits = mapping(report.get("by_split"), "by_split")
    for arm in totals:
        totals_match(published_totals.get(arm), totals[arm], f"totals.{arm}")
        for split in splits:
            totals_match(mapping(published_splits.get(split), split).get(arm), splits[split][arm],
                         f"by_split.{split}.{arm}")
    population = mapping(report.get("population"), "population")
    for key, count in {"cases": 9, "development": 6, "holdout": 3,
                       "missing_knowledge": 2, "positive": 7}.items():
        same(population.get(key), count, f"population.{key}")
    require(totals["candidate"]["response_chars"] < totals["baseline"]["response_chars"],
            "aggregate full-chain character savings absent")
    require(savings_cases >= 2, "savings are supported by only one case")
    acceptance = mapping(report.get("acceptance"), "acceptance")
    for key in ("aggregate_chars_reduced", "all_required_knowledge_preserved",
                "measured_cost_claim_passes", "no_forced_lookup_cost_increase",
                "no_mandatory_calls_added_on_sufficient_cases", "ranking_identical"):
        same(acceptance.get(key), True, f"acceptance.{key}")
    oracle = mapping(report.get("independent_oracle"), "independent_oracle")
    same(oracle.get("all_cases_pass"), True, "independent_oracle.all_cases_pass")
    same(oracle.get("all_positive_sufficient"), True, "independent_oracle.all_positive_sufficient")
    require(isinstance(oracle.get("loss_probe"), str) and oracle["loss_probe"].strip(),
            "independent oracle loss probe evidence missing")
    integer(oracle.get("targeted_tests_passed"), "independent_oracle.targeted_tests_passed", 1)
    require(isinstance(report.get("verdict"), str) and report["verdict"].startswith("PASS"),
            "verdict must state PASS after the recomputed checks")


def _verify(args: argparse.Namespace) -> None:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    _verify_report(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="hash-verified paired replay")
    for key in ("packet", "packet-sha256", "fixtures", "fixtures-sha256", "snapshot", "snapshot-sha256", "baseline-ref", "out"):
        run.add_argument("--" + key, required=True)
    run.add_argument("--candidate-ref", default="HEAD")
    run.add_argument("--aggregate-out")
    run.add_argument("--timeout", type=int, default=1800, help="seconds per arm")
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    for key in ("source-root", "packet", "snapshot", "snapshot-sha256", "out"):
        worker.add_argument("--" + key, required=True)
    verify = sub.add_parser("verify", help="check a published aggregate report without replay")
    verify.add_argument("--report", required=True, help="published aggregate report JSON")
    args = parser.parse_args(argv)
    try:
        if args.command == "_worker":
            _worker(args)
        elif args.command == "verify":
            _verify(args)
        else:
            _run(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
