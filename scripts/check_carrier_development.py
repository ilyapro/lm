#!/usr/bin/env python3
"""Local, development-only carrier replay using the unchanged frozen evaluator.

The full private packet is read only after all preregistered input hashes pass.
Only the uniquely selected development carrier enters the temporary packet.
Detailed evaluator reports stay in temporary scratch and are never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
PREREG = ROOT / "artifacts/recall-needed-cost/prereg.md"
FIXTURES = ROOT / "artifacts/recall-needed-cost/fixtures.json"
PACKET = Path("/home/sfx/p/ae/artifacts/recall-needed-cost/private/cases.json")
SNAPSHOT = Path("/home/sfx/p/ae/artifacts/recall-needed-cost/private/snapshot.sqlite3")
BASELINE = "cd4ad6c3984c41badc55199299532199ed4579b1"
REJECTED = "273f5a9e1385e292f20a3dde284fa9356047baab"
HASH_LABELS = {
    "Synthetic fixtures": FIXTURES,
    "Private case packet": PACKET,
    "Read only sfx snapshot": SNAPSHOT,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs() -> dict[str, str]:
    prereg = PREREG.read_text(encoding="utf-8")
    hashes: dict[str, str] = {}
    for label, path in HASH_LABELS.items():
        match = re.search(r"^\| " + re.escape(label) + r" \|[^\n]*\| `([0-9a-f]{64})` \|$", prereg, re.M)
        if match is None:
            raise ValueError(f"missing preregistered hash: {label}")
        expected = match.group(1)
        if sha256(path) != expected:
            raise ValueError(f"frozen hash mismatch: {label}")
        hashes[label] = expected
    return hashes


def selected_packet(packet: dict) -> dict:
    cases = [case for case in packet["cases"]
             if case.get("split") == "development" and case.get("family") == "known_large_carrier"]
    if len(cases) != 1:
        raise ValueError("expected exactly one development known_large_carrier")
    return {**packet, "cases": cases}


def revision(ref: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        text=True, timeout=15,
    ).strip()


def check_report(report: dict, *, baseline: str, candidate: str, rejected: bool) -> dict:
    if report.get("revisions") != {"baseline": baseline, "candidate": candidate}:
        raise ValueError("revision provenance mismatch")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        raise ValueError("single-case replay was skipped")
    case = cases[0]
    if case.get("split") != "development" or case.get("ranking_changed") is not False:
        raise ValueError("split or ordered retrieval IDs changed")
    before, after = case["baseline"], case["candidate"]
    if before["ordered_retrieval_ids"] != after["ordered_retrieval_ids"]:
        raise ValueError("ordered retrieval IDs changed")
    for row in (before, after):
        responses = row["responses"]
        if (not row["oracle_pass"] or not row["sufficient"] or row["remaining_requirements"]
                or not responses or responses[0]["tool"] != "memory_recall"
                or row["calls"] != len(responses)
                or row["recall_calls"] != sum(x["tool"] == "memory_recall" for x in responses)
                or row["lookup_calls"] != sum(x["tool"] == "memory_lookup" for x in responses)
                or row["response_chars"] != sum(x["chars"] for x in responses)):
            raise ValueError("insufficient knowledge, skipped execution or incomplete call accounting")
    if before["calls"] != 1 or before["lookup_calls"] != 0:
        raise ValueError("accepted baseline changed")
    if rejected:
        if after["calls"] != 2 or after["lookup_calls"] != 1 or after["response_chars"] <= before["response_chars"]:
            raise ValueError("rejected revision did not reproduce its costly lookup")
    elif after["calls"] != 1 or after["lookup_calls"] != 0 or after["response_chars"] >= before["response_chars"]:
        raise ValueError("candidate adds a call or fails to reduce full-chain characters")
    return {"baseline": before["response_chars"], "tested": after["response_chars"],
            "tested_calls": after["calls"], "tested_lookups": after["lookup_calls"]}


def replay(packet: Path, packet_hash: str, hashes: dict[str, str], candidate: str, out: Path) -> dict:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("LM_DELIVERY_") or key.startswith(("LM_RECALL_NEAR_DUP_", "LM_RECALL_GATE_", "LM_RECALL_REPEAT_")):
            env.pop(key)
    command = [sys.executable, str(ROOT / "scripts/recall_needed_cost.py"), "run",
               "--packet", str(packet), "--packet-sha256", packet_hash,
               "--snapshot", str(SNAPSHOT), "--snapshot-sha256", hashes["Read only sfx snapshot"],
               "--fixtures", str(FIXTURES), "--fixtures-sha256", hashes["Synthetic fixtures"],
               "--baseline-ref", BASELINE, "--candidate-ref", candidate,
               "--out", str(out), "--timeout", "180"]
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=420)
    out.with_suffix(".log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode or not out.is_file():
        raise RuntimeError("frozen evaluator failed or timed out; see local scratch for diagnostics")
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-ref", required=True)
    args = parser.parse_args()
    try:
        hashes = verify_inputs()
        packet = json.loads(PACKET.read_text(encoding="utf-8"))
        if packet.get("snapshot_sha256") != hashes["Read only sfx snapshot"]:
            raise ValueError("packet snapshot provenance mismatch")
        subset = selected_packet(packet)
        base, bad, candidate = revision(BASELINE), revision(REJECTED), revision(args.candidate_ref)
        if len({base, bad, candidate}) != 3:
            raise ValueError("candidate must differ from accepted and rejected revisions")
        scratch = Path(tempfile.mkdtemp(prefix="carrier-development-"))
        subset_path = scratch / "single-case.json"
        subset_path.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
        subset_hash = sha256(subset_path)
        rejected_report = replay(subset_path, subset_hash, hashes, bad, scratch / "rejected.json")
        candidate_report = replay(subset_path, subset_hash, hashes, candidate, scratch / "candidate.json")
        rejected_summary = check_report(rejected_report, baseline=base, candidate=bad, rejected=True)
        candidate_summary = check_report(candidate_report, baseline=base, candidate=candidate, rejected=False)
        if rejected_report["packet_sha256"] != subset_hash or candidate_report["packet_sha256"] != subset_hash:
            raise ValueError("subset provenance mismatch")
        if any(report.get("snapshot_sha256") != hashes["Read only sfx snapshot"]
               or report.get("fixtures_sha256") != hashes["Synthetic fixtures"]
               for report in (rejected_report, candidate_report)):
            raise ValueError("input provenance mismatch")
        if rejected_report["cases"][0]["baseline"]["ordered_retrieval_ids"] != candidate_report["cases"][0]["baseline"]["ordered_retrieval_ids"]:
            raise ValueError("baseline retrieval differs between isolated replays")
        print(f"frozen hashes: fixtures={hashes['Synthetic fixtures']} packet={hashes['Private case packet']} snapshot={hashes['Read only sfx snapshot']}")
        print(f"revisions: baseline={base} rejected={bad} candidate={candidate}")
        print(f"development carrier: baseline={candidate_summary['baseline']} chars/1 call; rejected={rejected_summary['tested']} chars/{rejected_summary['tested_calls']} calls/{rejected_summary['tested_lookups']} lookup; candidate={candidate_summary['tested']} chars/{candidate_summary['tested_calls']} call/0 lookup; ranking unchanged")
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"carrier development check failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
