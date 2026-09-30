"""Run the frozen holdout of holdout-protocol.md on a private copy; aggregates only.

Usage: python3 holdout.py <copy.sqlite3> <work-dir>
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import replay as R  # noqa: E402
from living_memory import consolidation as C  # noqa: E402
from living_memory.server import create_mcp_server  # noqa: E402
from living_memory.storage import MemoryStore  # noqa: E402
from test_transport_identity import FakeMCP  # noqa: E402

SEED = 20261001
SAMPLE = 40
VIEWED = 25


def supersedes_sources(store: MemoryStore) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = defaultdict(set)
    for source, target in store.connection.execute(
        "SELECT source_id, target_id FROM connections WHERE type='supersedes'"
    ):
        edges[source].add(target)
    return edges


def heads_order(store: MemoryStore, edges: dict[str, set[str]]) -> list[tuple[str, str]]:
    """Groups by teach-head count, in the order the repair session displayed them."""

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for node in store.list_nodes(level="trace", include_decayed=True, limit=10**7):
        if key := C._procedure_key(node):
            groups[(node.scope, key.group_id)].append(node)
    rows = []
    for group, members in groups.items():
        if sum(1 for node in members if not node.decayed) < 3:
            continue
        ids = {node.id for node in members}
        live = [node for node in members if not node.decayed and not C._is_superseded(store, node.id)]
        rows.append((group, sum(1 for node in live if edges.get(node.id, set()) & ids)))
    return [group for group, _heads in sorted(rows, key=lambda row: -row[1])]


def holdout(store: MemoryStore, stratum: bool = False) -> list[dict]:
    edges = supersedes_sources(store)
    design = R.eval_sets(store)
    excluded = set(heads_order(store, edges)[:VIEWED])
    for scope, query, target in design["schema_topics"]:
        trigger = query.removeprefix("how to ")
        excluded |= {
            (schema.scope, C._schema_group_id(schema))
            for schema in store.list_nodes(level="schema", scope=scope, include_decayed=False, limit=100_000)
            if str(schema.context.get("trigger") or "") == trigger
        }
    for scope, _query, target in design["prose_recipes"]:
        excluded.add((scope, C._procedure_key(store.get_node(target)).group_id))
    population = []
    for schema in store.list_nodes(level="schema", include_decayed=False, limit=1_000_000):
        trigger = str(schema.context.get("trigger") or "")
        group = (schema.scope, C._schema_group_id(schema))
        if schema.provenance.get("strategy") != "procedural" or not R.readable(trigger):
            continue
        if group not in excluded:
            population.append(schema)
    population.sort(key=lambda schema: schema.id)
    random.Random(SEED).shuffle(population)
    cases = []
    for schema in population if stratum else population[:SAMPLE]:
        items = []
        for node_id in schema.source_traces:
            node = store.get_node(node_id)
            if node is None or node.decayed or C._is_superseded(store, node_id):
                continue
            if node.content.strip() not in schema.content:
                continue
            instruction = bool(R.RECIPE_HEAD.match(node.content)) or bool(edges.get(node_id))
            items.append({"id": node_id, "text": node.content, "instruction": instruction})
        if stratum and not any(item["instruction"] for item in items):
            continue
        cases.append({"scope": schema.scope, "query": f"how to {schema.context['trigger']}", "items": items})
    return cases


def measure(db: Path, cases: list[dict], unconfirmed: dict[str, int]) -> dict:
    mcp = create_mcp_server(db, mcp_factory=FakeMCP)
    lookup, recall = mcp.tools["memory_lookup"], mcp.tools["memory_recall"]
    chars = forced = delivered_unconfirmed = 0
    obtained: list[list[bool]] = []
    for case in cases:
        response = recall(case["query"], scope=case["scope"], max_results=10)
        chars += len(json.dumps(response, ensure_ascii=False))
        texts = []
        for result in response["results"]:
            node = result["node"]
            if node.get("level") == "schema":
                delivered_unconfirmed += unconfirmed.get(node["id"], 0)
            text, cost = R.read(result, lookup)
            forced += cost
            texts.append(text)
        joined = "\n".join(texts)
        obtained.append([item["text"] in joined for item in case["items"]])
    mcp.memory_store.close()
    return {"chars": chars + forced, "forced": forced, "unconfirmed_delivered": delivered_unconfirmed, "obtained": obtained}


def main() -> None:
    source, work = Path(sys.argv[1]), Path(sys.argv[2])
    R.load_operator_env()
    work.mkdir(parents=True, exist_ok=True)
    before_db, after_db = work / "before.sqlite3", work / "after.sqlite3"
    shutil.copyfile(source, before_db)
    shutil.copyfile(source, after_db)
    with MemoryStore(before_db) as store:
        cases = holdout(store, stratum=sys.argv[3:] == ["stratum"])
        census_before = R.census(store)
    with MemoryStore(after_db) as store:
        R.procedural_pass(store)
        R.procedural_pass(store)
        census_after = R.census(store)
    was = measure(before_db, cases, census_before["_unconfirmed"])
    now = measure(after_db, cases, census_after["_unconfirmed"])
    tally = defaultdict(int)
    for case, before_row, after_row in zip(cases, was["obtained"], now["obtained"]):
        for item, a, b in zip(case["items"], before_row, after_row):
            kind = "instruction" if item["instruction"] else "case"
            tally[f"{kind}_items"] += 1
            tally[f"{kind}_{'kept' if a and b else 'lost' if a else 'gained' if b else 'baseline_miss'}"] += 1
    report = {
        "groups": len(cases),
        "items": dict(sorted(tally.items())),
        "H1_new_instruction_losses": tally["instruction_lost"],
        "H2_census_unconfirmed_after": census_after["unconfirmed_steps"],
        "H2_unconfirmed_delivered_after": now["unconfirmed_delivered"],
        "H3_chars_before_after": [was["chars"], now["chars"]],
        "unconfirmed_delivered_before": was["unconfirmed_delivered"],
        "forced_lookup_chars_before_after": [was["forced"], now["forced"]],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
