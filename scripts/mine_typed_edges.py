#!/usr/bin/env python3
"""Read-only miner for provenance-derived typed-edge candidate rules.

Mines the Living Memory corpus for candidate typed, directional edges derivable
from structured provenance (teach lineage, failure->resolution context labels,
procedure groups, shared commit/files, concept/schema source_traces) and
measures, per rule:

  * derivable candidate count (directed) and unique undirected pairs,
  * both-active pair count (graph traversal skips decayed nodes,
    src/living_memory/retrieval.py:525, so only both-active edges add reach),
  * overlap with the existing edge inventory (any type, either direction) ->
    new-information fraction,
  * co-retrieval redundancy: fraction of pairs that already co-occurred in a
    recorded recall_events result list (bm25/vector already surface them
    together; lower = more non-redundant graph signal),
  * a seeded random sample of pairs for manual precision inspection.

The DB is opened strictly read-only (sqlite URI mode=ro + PRAGMA query_only).
No writes are performed anywhere. Findings are consumed by
artifacts/discovery/typed-edge-rules.md.

Usage:
  python3 scripts/mine_typed_edges.py --db /path/to/global.sqlite3 \
      --report /tmp/typed_edge_report.json --samples /tmp/typed_edge_samples.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable

DEFAULT_DB = "/home/sfx/.local/share/living-memory/global.sqlite3"

# ULID node ids as they appear verbatim inside trace content (Crockford base32).
ULID_RE = re.compile(r"\b01[0-9A-HJKMNP-TV-Z]{24}\b")

# Adjacent correction/update phrasing shortly before a named node ULID.
# Catches "Correction for trace 01K...", "UPDATE/correction to node 01K...",
# "corrects/supersedes the leading hypothesis in 01K...". A loose 80-char
# any-correction-word window scored 8-9/12 sampled precision; the adjacent
# form below drops the pairs where the trigger word referred to a third node.
CORRECTION_NEAR_ULID_RE = re.compile(
    r"(?i)(?:\b(?:correction|update)s?(?:/\w+)?[-\s]+(?:to|of|for)\b|\b(?:corrects|supersedes)\b)"
    r"[^\n]{0,60}?\b(01[0-9A-HJKMNP-TV-Z]{24})\b"
)

# --- rule label sets (grounded in observed context values on the live corpus) ---

CORRECTION_CONTEXT_TYPES = {"correction", "memory_correction", "decomposition_correction"}
CORRECTION_CONTENT_PREFIXES = ("Correction:", "CORRECTION:", "Correction ")

PROBLEM_TYPES = {
    "root_cause",
    "root_cause_and_debugging_insights",
    "failure",
    "error",
    "bug",
    "blocker",
    "incident",
    "regression",
    "verification_finding",
}
PROBLEM_LESSON_KINDS = {
    "implementation_bug",
    "race_condition",
    "missing_dependency",
    "execution_blocker",
    "spec_ambiguity",
    "root_cause",
}
PROBLEM_OUTCOMES = {"structural_fail", "reopened", "fail", "failure", "error"}

SOLUTION_TYPES = {
    "failure_resolution",
    "working_fix",
    "fix",
    "gate_failure_resolution",
    "supervision_repair",
}
SOLUTION_LESSON_KINDS = {
    "tree_decomposition_repair",
    "decomposition_repair",
    "decomposition-repair",
    "parent-decomposition-repair",
    "working_fix",
}

ROOT_CAUSE_TYPES = {"root_cause", "root_cause_and_debugging_insights"}
FAILURE_EVIDENCE_TYPES = {"verification_finding", "failure", "error", "regression"}


@dataclass
class NodeRow:
    id: str
    level: str
    scope: str
    decayed: bool
    timestamp: str
    content: str
    context: dict[str, Any]
    source_traces: list[str]
    provenance: dict[str, Any]

    @property
    def ts(self) -> dt.datetime | None:
        return parse_ts(self.timestamp or self.context.get("timestamp") or "")


@dataclass
class Candidate:
    """One derived directed edge candidate."""

    rule: str
    src: str
    dst: str
    type: str
    directed: bool
    note: str = ""

    @property
    def pair(self) -> tuple[str, str]:
        return (self.src, self.dst) if self.src <= self.dst else (self.dst, self.src)


@dataclass
class RuleReport:
    rule: str
    proposed_type: str
    directed: bool
    candidates_directed: int = 0
    unique_pairs: int = 0
    both_active_pairs: int = 0
    pairs_with_decayed_endpoint: int = 0
    overlap_any_edge: int = 0
    overlap_same_type: int = 0
    co_retrieved: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        active = self.both_active_pairs
        return {
            "rule": self.rule,
            "proposed_type": self.proposed_type,
            "directed": self.directed,
            "candidates_directed": self.candidates_directed,
            "unique_pairs": self.unique_pairs,
            "both_active_pairs": active,
            "pairs_with_decayed_endpoint": self.pairs_with_decayed_endpoint,
            "overlap_any_edge_active_pairs": self.overlap_any_edge,
            "overlap_same_type_active_pairs": self.overlap_same_type,
            "new_information_fraction": round(1 - self.overlap_any_edge / active, 4) if active else None,
            "co_retrieved_active_pairs": self.co_retrieved,
            "co_retrieval_redundancy": round(self.co_retrieved / active, 4) if active else None,
            "notes": self.notes,
        }


def parse_ts(value: str) -> dt.datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def load_json(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def open_ro(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    return con


def load_corpus(con: sqlite3.Connection) -> dict[str, NodeRow]:
    nodes: dict[str, NodeRow] = {}
    query = (
        "SELECT id, level, scope, decayed, timestamp, content, context,"
        " source_traces, provenance FROM nodes"
    )
    for row in con.execute(query):
        nodes[row["id"]] = NodeRow(
            id=row["id"],
            level=row["level"],
            scope=row["scope"],
            decayed=bool(row["decayed"]),
            timestamp=row["timestamp"] or "",
            content=row["content"] or "",
            context=load_json(row["context"], {}),
            source_traces=[str(x) for x in load_json(row["source_traces"], [])],
            provenance=load_json(row["provenance"], {}),
        )
    return nodes


def load_edges(con: sqlite3.Connection) -> list[tuple[str, str, str, dict[str, Any]]]:
    edges = []
    for row in con.execute("SELECT source_id, target_id, type, metadata FROM connections"):
        edges.append((row["source_id"], row["target_id"], row["type"], load_json(row["metadata"], {})))
    return edges


def load_coretrieval_pairs(con: sqlite3.Connection) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in con.execute("SELECT results FROM recall_events"):
        results = load_json(row["results"], [])
        ids = sorted({str(item.get("node_id")) for item in results if isinstance(item, dict) and item.get("node_id")})
        for a, b in combinations(ids, 2):
            pairs.add((a, b))
    return pairs


def is_rejected_alternative(node: NodeRow) -> bool:
    return bool(node.context.get("is_rejected_alternative"))


def is_problem(node: NodeRow) -> bool:
    ctx = node.context
    return (
        str(ctx.get("type") or "") in PROBLEM_TYPES
        or str(ctx.get("lesson_kind") or "") in PROBLEM_LESSON_KINDS
        or str(ctx.get("outcome") or "") in PROBLEM_OUTCOMES
    )


def is_solution(node: NodeRow) -> bool:
    ctx = node.context
    return (
        str(ctx.get("type") or "") in SOLUTION_TYPES
        or str(ctx.get("lesson_kind") or "") in SOLUTION_LESSON_KINDS
    )


# --- rules -------------------------------------------------------------------


def rule_supersedes_transitive(
    nodes: dict[str, NodeRow], edges: list[tuple[str, str, str, dict[str, Any]]]
) -> list[Candidate]:
    """R1a: transitive closure over existing supersedes chains (A=>B=>C -> A=>C)."""

    adj: dict[str, set[str]] = defaultdict(set)
    direct = set()
    for src, dst, etype, _ in edges:
        if etype == "supersedes":
            adj[src].add(dst)
            direct.add((src, dst))
    out: list[Candidate] = []
    for start in list(adj):
        reach: set[str] = set()
        stack = list(adj[start])
        while stack:
            cur = stack.pop()
            if cur in reach:
                continue
            reach.add(cur)
            stack.extend(adj.get(cur, ()))
        for end in reach:
            if end != start and (start, end) not in direct:
                out.append(Candidate("R1a_supersedes_transitive", start, end, "supersedes", True, "chain closure"))
    return out


def _is_correction_marked(node: NodeRow) -> bool:
    ctx_type = str(node.context.get("type") or "")
    lesson_kind = str(node.context.get("lesson_kind") or "")
    return (
        ctx_type in CORRECTION_CONTEXT_TYPES
        or lesson_kind == "correction"
        or node.content.startswith(CORRECTION_CONTENT_PREFIXES)
    )


def rule_correction_lineage(nodes: dict[str, NodeRow]) -> list[Candidate]:
    """R1b: correction-marked remember-trace supersedes a same-task recalled node.

    memory_teach already writes supersedes edges, but corrections written via
    memory_remember get only generic implicit 'related' edges. Sampling showed
    latest-recall fallbacks are noise, so targets are restricted to recalled
    nodes (provenance.prior_recalls result ids) sharing the same (scope, task).
    Corrections that name their target ULID in content are handled by R1c.
    """

    out: list[Candidate] = []
    for node in nodes.values():
        if node.decayed or node.level != "trace" or not _is_correction_marked(node):
            continue
        if ULID_RE.search(node.content):
            continue  # explicit target -> R1c
        prior = node.provenance.get("prior_recalls") or []
        task = str(node.context.get("task") or "")
        if not isinstance(prior, list) or not prior or not task:
            continue
        recalled: list[str] = []
        for entry in prior:
            if isinstance(entry, dict):
                recalled.extend(str(x) for x in entry.get("result_ids") or [])
        same_task = [
            rid
            for rid in dict.fromkeys(recalled)
            if rid != node.id
            and rid in nodes
            and nodes[rid].scope == node.scope
            and str(nodes[rid].context.get("task") or "") == task
        ]
        for rid in same_task[:2]:
            out.append(Candidate("R1b_correction_lineage", node.id, rid, "supersedes", True, "recalled + same task"))
    return out


def _correction_ulid_targets(node: NodeRow) -> list[str]:
    hits = [m.group(1) for m in CORRECTION_NEAR_ULID_RE.finditer(node.content)]
    if not hits and _is_correction_marked(node):
        hits = ULID_RE.findall(node.content)
    return [u for u in dict.fromkeys(hits) if u != node.id]


def rule_content_ulid_supersedes(nodes: dict[str, NodeRow]) -> list[Candidate]:
    """R1c: trace whose content names a corrected/updated node ULID supersedes it."""

    out: list[Candidate] = []
    for node in nodes.values():
        if node.decayed or node.level != "trace":
            continue
        for ulid in _correction_ulid_targets(node):
            if ulid in nodes:
                out.append(
                    Candidate("R1c_content_ulid_supersedes", node.id, ulid, "supersedes", True, "target named in content")
                )
    return out


def rule_content_ulid_reference(nodes: dict[str, NodeRow]) -> list[Candidate]:
    """R6: trace naming another node's ULID references it (typed related).

    Pairs already claimed by R1c (correction-word near the ULID) are excluded.
    """

    out: list[Candidate] = []
    for node in nodes.values():
        if node.decayed or node.level != "trace":
            continue
        correction_targets = set(_correction_ulid_targets(node))
        for ulid in dict.fromkeys(ULID_RE.findall(node.content)):
            if ulid != node.id and ulid in nodes and ulid not in correction_targets:
                out.append(
                    Candidate("R6_content_ulid_reference", node.id, ulid, "related", True, "explicit id reference")
                )
    return out


def _grouped_by(nodes: dict[str, NodeRow], key: str) -> dict[tuple[str, str], list[NodeRow]]:
    groups: dict[tuple[str, str], list[NodeRow]] = defaultdict(list)
    for node in nodes.values():
        if node.decayed or node.level != "trace":
            continue
        value = node.context.get(key)
        if value:
            groups[(node.scope, str(value))].append(node)
    return groups


RESOLUTION_WINDOW_SECONDS = 3600


def _is_tight_problem(node: NodeRow) -> bool:
    ctx = node.context
    return (
        str(ctx.get("type") or "") in ROOT_CAUSE_TYPES
        or str(ctx.get("outcome") or "") in {"structural_fail", "reopened"}
    )


def rule_failure_resolution(nodes: dict[str, NodeRow]) -> list[Candidate]:
    """R2a: root-cause trace 'caused'-links to the resolution written just after it.

    Sampling the loose variant (any problem label, any gap, same task) gave
    ~0.2 precision: positive-polarity verification_finding and self-contained
    implementation_bug lessons leaked in, and recurring umbrella tasks paired
    unrelated incidents. All true pairs sat <=4 minutes apart and the one
    false pair at a 6h window sat 287 minutes out, so the tight rule keeps
    root_cause-typed (or structural_fail/reopened-outcome) problems only,
    pairs within task OR session groups, requires the resolution to follow
    within RESOLUTION_WINDOW_SECONDS, and links the nearest one.

    Direction root_cause -> resolution as 'caused': backward traversal from a
    resolution finds its cause (0.85 / 1.0 causal), forward from a matched
    root-cause text surfaces the fix (0.75), and the pair becomes reachable in
    causal mode, which follows only caused/requires — a channel that currently
    has zero edges. 'supersedes' was rejected: causal mode skips it and the
    superseded rank penalty would wrongly punish still-valid root-cause
    knowledge.
    """

    best: dict[tuple[str, str], tuple[float, Candidate]] = {}
    for key in ("task", "session_id"):
        for _, members in sorted(_grouped_by(nodes, key).items()):
            members = [m for m in members if not is_rejected_alternative(m)]
            problems = [m for m in members if _is_tight_problem(m) and not is_solution(m)]
            solutions = [m for m in members if is_solution(m)]
            if not problems or not solutions:
                continue
            for sol in solutions:
                sol_ts = sol.ts
                if sol_ts is None:
                    continue
                scored: list[tuple[float, NodeRow]] = []
                for prob in problems:
                    prob_ts = prob.ts
                    if prob.id == sol.id or prob_ts is None:
                        continue
                    delta = (sol_ts - prob_ts).total_seconds()
                    if 0 <= delta <= RESOLUTION_WINDOW_SECONDS:
                        scored.append((delta, prob))
                if not scored:
                    continue
                delta, prob = min(scored, key=lambda item: (item[0], item[1].id))
                cand = Candidate(
                    "R2a_root_cause_caused_resolution",
                    prob.id,
                    sol.id,
                    "caused",
                    True,
                    f"prob={prob.context.get('type') or prob.context.get('outcome')}"
                    f" sol={sol.context.get('type') or sol.context.get('lesson_kind')}"
                    f" gap_min={round(delta / 60)} via={key}",
                )
                current = best.get((prob.id, sol.id))
                if current is None or delta < current[0]:
                    best[(prob.id, sol.id)] = (delta, cand)
    return [cand for _, cand in best.values()]


def rule_root_cause_caused(nodes: dict[str, NodeRow]) -> list[Candidate]:
    """R2b: root_cause-labeled trace 'caused'-links to failure-evidence trace, same task."""

    out: list[Candidate] = []
    for _, members in sorted(_grouped_by(nodes, "task").items()):
        members = [m for m in members if not is_rejected_alternative(m)]
        causes = [m for m in members if str(m.context.get("type") or "") in ROOT_CAUSE_TYPES]
        effects = [
            m
            for m in members
            if str(m.context.get("type") or "") in FAILURE_EVIDENCE_TYPES
            or str(m.context.get("outcome") or "") in {"structural_fail", "reopened"}
        ]
        if not causes or not effects:
            continue
        for cause in causes:
            cause_ts = cause.ts
            scored: list[tuple[float, NodeRow]] = []
            for effect in effects:
                if effect.id == cause.id:
                    continue
                effect_ts = effect.ts
                gap = abs((cause_ts - effect_ts).total_seconds()) if cause_ts and effect_ts else float("inf")
                scored.append((gap, effect))
            scored.sort(key=lambda item: (item[0], item[1].id))
            for _, effect in scored[:2]:
                out.append(
                    Candidate(
                        "R2b_root_cause_caused",
                        cause.id,
                        effect.id,
                        "caused",
                        True,
                        f"effect={effect.context.get('type') or effect.context.get('outcome')}",
                    )
                )
    return out


def rule_procedure_requires(nodes: dict[str, NodeRow], max_group: int = 12) -> list[Candidate]:
    """R3: consecutive timestamp-ordered traces of one procedure_id chain via 'requires'."""

    out: list[Candidate] = []
    for _, members in sorted(_grouped_by(nodes, "procedure_id").items()):
        members = [m for m in members if not is_rejected_alternative(m)]
        if not 2 <= len(members) <= max_group:
            continue
        stamps = {m.timestamp for m in members}
        if len(stamps) < 2:
            continue  # single batch write, no order information
        ordered = sorted(members, key=lambda m: (m.timestamp, m.id))
        for earlier, later in zip(ordered, ordered[1:]):
            out.append(
                Candidate(
                    "R3_procedure_requires",
                    later.id,
                    earlier.id,
                    "requires",
                    True,
                    f"procedure_id={members[0].context.get('procedure_id')}",
                )
            )
    return out


def rule_same_commit(nodes: dict[str, NodeRow], max_group: int = 10) -> list[Candidate]:
    """R4a: active nodes recorded against the same commit (7-char prefix match)."""

    groups: dict[str, list[NodeRow]] = defaultdict(list)
    for node in nodes.values():
        if node.decayed:
            continue
        commit = str(node.context.get("commit") or "").strip().lower()
        if len(commit) >= 7:
            groups[commit[:7]].append(node)
    out: list[Candidate] = []
    for commit, members in sorted(groups.items()):
        if not 2 <= len(members) <= max_group:
            continue
        for a, b in combinations(sorted(members, key=lambda m: m.id), 2):
            if is_rejected_alternative(a) and is_rejected_alternative(b):
                continue  # RA siblings already hang off the chosen node
            out.append(Candidate("R4a_same_commit", a.id, b.id, "related", False, f"commit={commit}"))
    return out


def rule_shared_files(nodes: dict[str, NodeRow], max_fanout: int = 20) -> list[Candidate]:
    """R4b: active nodes citing the same context.files entry (fanout-capped)."""

    postings: dict[str, list[str]] = defaultdict(list)
    for node in nodes.values():
        if node.decayed:
            continue
        files = node.context.get("files")
        if isinstance(files, list):
            for f in files:
                postings[str(f)].append(node.id)
    pair_files: dict[tuple[str, str], list[str]] = defaultdict(list)
    for fname, ids in sorted(postings.items()):
        if not 2 <= len(ids) <= max_fanout:
            continue
        for a, b in combinations(sorted(set(ids)), 2):
            if is_rejected_alternative(nodes[a]) and is_rejected_alternative(nodes[b]):
                continue
            pair_files[(a, b)].append(fname)
    out: list[Candidate] = []
    for (a, b), files in sorted(pair_files.items()):
        out.append(
            Candidate("R4b_shared_files", a, b, "related", False, f"shared={len(files)}:{files[0]}")
        )
    return out


def rule_derived_from(nodes: dict[str, NodeRow]) -> list[Candidate]:
    """R5a: schema -> member source trace (provenance ground truth).

    Schemas group traces by exact task_pattern/procedure_id fields, so the
    derived-from relation is coherent at any size. Concepts are excluded:
    similarity clustering produced mega-clusters (up to 2,209 sources) whose
    representative content is not an abstraction of arbitrary members —
    sampled concept pairs scored 0/16 as derived-from.
    """

    out: list[Candidate] = []
    for node in nodes.values():
        if node.decayed or node.level != "schema":
            continue
        size = len(node.source_traces)
        for trace_id in node.source_traces:
            if trace_id in nodes:
                out.append(
                    Candidate(
                        "R5a_derived_from",
                        node.id,
                        trace_id,
                        "related",  # proposed metadata.kind=derived_from on base type related
                        True,
                        f"{node.level} sources={size}",
                    )
                )
    return out


def rule_trace_informed_by(nodes: dict[str, NodeRow]) -> list[Candidate]:
    """R5b: trace -> trace-level source_traces (nodes recalled just before writing)."""

    out: list[Candidate] = []
    for node in nodes.values():
        if node.decayed or node.level != "trace":
            continue
        for trace_id in node.source_traces:
            if trace_id in nodes and trace_id != node.id:
                out.append(Candidate("R5b_trace_informed_by", node.id, trace_id, "related", True, "informed-by"))
    return out


# --- measurement -------------------------------------------------------------


def summarize_rule(
    rule: str,
    proposed_type: str,
    directed: bool,
    candidates: list[Candidate],
    nodes: dict[str, NodeRow],
    existing_pairs: set[tuple[str, str]],
    existing_typed_pairs: dict[str, set[tuple[str, str]]],
    coretrieved: set[tuple[str, str]],
) -> tuple[RuleReport, list[Candidate]]:
    report = RuleReport(rule=rule, proposed_type=proposed_type, directed=directed)
    report.candidates_directed = len(candidates)
    by_pair: dict[tuple[str, str], Candidate] = {}
    for cand in candidates:
        by_pair.setdefault(cand.pair, cand)
    report.unique_pairs = len(by_pair)
    active: list[Candidate] = []
    same_type = existing_typed_pairs.get(proposed_type, set())
    for pair, cand in by_pair.items():
        a, b = pair
        node_a, node_b = nodes.get(a), nodes.get(b)
        if not node_a or not node_b or node_a.decayed or node_b.decayed:
            report.pairs_with_decayed_endpoint += 1
            continue
        active.append(cand)
        if pair in existing_pairs:
            report.overlap_any_edge += 1
        if pair in same_type:
            report.overlap_same_type += 1
        if pair in coretrieved:
            report.co_retrieved += 1
    report.both_active_pairs = len(active)
    return report, active


def excerpt(text: str, limit: int = 360) -> str:
    flat = " ".join((text or "").split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def context_line(node: NodeRow) -> str:
    ctx = node.context
    bits = []
    for key in ("type", "lesson_kind", "outcome", "task", "procedure_id", "commit", "session_id"):
        value = ctx.get(key)
        if value:
            bits.append(f"{key}={str(value)[:60]}")
    return " ".join(bits)


def render_samples(
    rule: str,
    sampled: list[Candidate],
    nodes: dict[str, NodeRow],
    existing_pairs: set[tuple[str, str]],
    coretrieved: set[tuple[str, str]],
) -> list[str]:
    lines = [f"\n## {rule} — {len(sampled)} sampled pairs\n"]
    for index, cand in enumerate(sampled, start=1):
        src, dst = nodes[cand.src], nodes[cand.dst]
        pair = cand.pair
        lines.append(
            f"### {rule} #{index}: `{cand.src}` -[{cand.type}]-> `{cand.dst}`"
            f"  (already-edged={pair in existing_pairs}, co-retrieved={pair in coretrieved})"
        )
        lines.append(f"- note: {cand.note}")
        lines.append(f"- SRC [{src.level} {src.scope} {src.timestamp}] {context_line(src)}")
        lines.append(f"  > {excerpt(src.content)}")
        lines.append(f"- DST [{dst.level} {dst.scope} {dst.timestamp}] {context_line(dst)}")
        lines.append(f"  > {excerpt(dst.content)}")
        lines.append("- verdict: PENDING")
        lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help="path to global.sqlite3 (opened read-only)")
    parser.add_argument("--report", default="/tmp/typed_edge_report.json")
    parser.add_argument("--samples", default="/tmp/typed_edge_samples.md")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-file-fanout", type=int, default=20)
    parser.add_argument("--max-proc-group", type=int, default=12)
    parser.add_argument("--max-commit-group", type=int, default=10)
    args = parser.parse_args(argv)

    con = open_ro(args.db)
    nodes = load_corpus(con)
    edges = load_edges(con)
    coretrieved = load_coretrieval_pairs(con)
    con.close()

    existing_pairs: set[tuple[str, str]] = set()
    existing_typed_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    edge_type_counts = Counter()
    active_pairs_existing = 0
    for src, dst, etype, _ in edges:
        pair = (src, dst) if src <= dst else (dst, src)
        if pair not in existing_pairs:
            src_node, dst_node = nodes.get(src), nodes.get(dst)
            if src_node and dst_node and not src_node.decayed and not dst_node.decayed:
                active_pairs_existing += 1
        existing_pairs.add(pair)
        existing_typed_pairs[etype].add(pair)
        edge_type_counts[etype] += 1

    active_nodes = sum(1 for n in nodes.values() if not n.decayed)
    rule_specs: list[tuple[str, str, bool, list[Candidate]]] = [
        ("R1a_supersedes_transitive", "supersedes", True, rule_supersedes_transitive(nodes, edges)),
        ("R1b_correction_lineage", "supersedes", True, rule_correction_lineage(nodes)),
        ("R1c_content_ulid_supersedes", "supersedes", True, rule_content_ulid_supersedes(nodes)),
        ("R2a_root_cause_caused_resolution", "caused", True, rule_failure_resolution(nodes)),
        ("R2b_root_cause_caused", "caused", True, rule_root_cause_caused(nodes)),
        ("R3_procedure_requires", "requires", True, rule_procedure_requires(nodes, args.max_proc_group)),
        ("R4a_same_commit", "related", False, rule_same_commit(nodes, args.max_commit_group)),
        ("R4b_shared_files", "related", False, rule_shared_files(nodes, args.max_file_fanout)),
        ("R5a_derived_from", "related", True, rule_derived_from(nodes)),
        ("R5b_trace_informed_by", "related", True, rule_trace_informed_by(nodes)),
        ("R6_content_ulid_reference", "related", True, rule_content_ulid_reference(nodes)),
    ]

    rng = random.Random(args.seed)
    reports: list[dict[str, Any]] = []
    sample_lines: list[str] = [
        "# Typed-edge rule samples (seeded, read-only mining)",
        f"db={args.db} seed={args.seed} sample_size={args.sample_size}",
        "Legend: already-edged = pair already connected by ANY existing edge;",
        "co-retrieved = pair appeared together in >=1 recorded recall_events result list.",
    ]
    for rule, proposed_type, directed, candidates in rule_specs:
        report, active = summarize_rule(
            rule, proposed_type, directed, candidates, nodes,
            existing_pairs, existing_typed_pairs, coretrieved,
        )
        reports.append(report.as_dict())
        pool = sorted(active, key=lambda c: (c.src, c.dst))
        sampled = pool if len(pool) <= args.sample_size else rng.sample(pool, args.sample_size)
        sample_lines.extend(render_samples(rule, sampled, nodes, existing_pairs, coretrieved))

    summary = {
        "db": args.db,
        "seed": args.seed,
        "nodes_total": len(nodes),
        "nodes_active": active_nodes,
        "edges_total": len(edges),
        "edge_type_counts": dict(edge_type_counts),
        "typed_edge_share_now": round(
            (edge_type_counts["caused"] + edge_type_counts["contradicts"]
             + edge_type_counts["supersedes"] + edge_type_counts["requires"]) / max(1, len(edges)),
            4,
        ),
        "coretrieved_pairs": len(coretrieved),
        "existing_unique_pairs": len(existing_pairs),
        "existing_unique_pairs_both_active": active_pairs_existing,
        "params": {
            "max_file_fanout": args.max_file_fanout,
            "max_proc_group": args.max_proc_group,
            "max_commit_group": args.max_commit_group,
            "sample_size": args.sample_size,
        },
        "rules": reports,
    }
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(args.samples, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sample_lines))
    json.dump(summary, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
