"""Holdout 3 of holdout-protocol.md (frozen before its first run); aggregates only.

  python3 holdout3.py items   <copy.sqlite3> <items.json>   # draw the set (current code)
  LM_SRC=<checkout>/src python3 holdout3.py measure <db> <items.json> <out.json>
  python3 holdout3.py pass    <db>                          # two ordinary procedural passes

``measure`` runs whatever code ``LM_SRC`` points at, so ``before`` is measured
with the original code on the untouched copy and ``after`` with the candidate
after its passes. items.json holds private ids and texts: it stays in scratch.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("LM_SRC") or HERE.parents[1] / "src")
sys.path[:0] = [str(SRC), str(HERE.parents[1] / "tests")]
import living_memory  # noqa: E402 (pin the code before replay changes sys.path)

assert Path(living_memory.__file__).resolve().is_relative_to(SRC.resolve()), living_memory.__file__

SEED = 20261002
SAMPLE = 60
COLLATERAL = 60
RECIPE_HEAD = re.compile(r"^\s*(procedure|процедура|recipe|рецепт)\b", re.IGNORECASE)


def items(copy: Path, out: Path) -> None:
    sys.path.insert(0, str(HERE))
    import holdout as H
    import replay as R
    from living_memory import consolidation as C
    from living_memory.storage import MemoryStore

    R.load_operator_env()
    with MemoryStore(copy) as store:
        edges = H.supersedes_sources(store)
        seen: set[tuple[str, str]] = set(H.heads_order(store, edges)[: H.VIEWED])
        design = R.eval_sets(store)
        for scope, query, target in design["schema_topics"] + design["prose_recipes"]:
            if target:
                seen.add((scope, C._procedure_key(store.get_node(target)).group_id))
            else:
                trigger = query.removeprefix("how to ")
                seen |= {
                    (s.scope, C._schema_group_id(s))
                    for s in store.list_nodes(level="schema", scope=scope, include_decayed=False, limit=100_000)
                    if str(s.context.get("trigger") or "") == trigger
                }
        for stratum in (False, True):
            for case in H.holdout(store, stratum=stratum):
                for item in case["items"]:
                    seen.add((case["scope"], C._procedure_key(store.get_node(item["id"])).group_id))
        eligible = []
        for node in store.list_nodes(level="trace", include_decayed=False, limit=10**7):
            key = C._procedure_key(node)
            if key is None or (node.scope, key.group_id) in seen or C._is_superseded(store, node.id):
                continue
            if not (RECIPE_HEAD.match(node.content) or edges.get(node.id)):
                continue
            name = C._normalize_trigger(key.raw)
            if R.readable(name):
                eligible.append((node.scope, key.group_id, node.id, f"how to {name}", node.content))
        eligible.sort(key=lambda row: row[2])
        random.Random(SEED).shuffle(eligible)
        chosen, groups = [], set()
        for scope, group, node_id, query, text in eligible:
            if (scope, group) not in groups and len(chosen) < SAMPLE:
                groups.add((scope, group))
                chosen.append({"scope": scope, "id": node_id, "query": query, "text": text})
        queries = sorted(
            {(row[0], row[1]) for row in store.connection.execute(
                "SELECT scope, query FROM recall_events WHERE scope IS NOT NULL AND length(query) > 12"
            )}
        )
        random.Random(SEED + 1).shuffle(queries)
        collateral = [{"scope": scope, "query": query} for scope, query in queries[:COLLATERAL]]
    json.dump({"eligible": len(eligible), "items": chosen, "collateral": collateral}, out.open("w"), ensure_ascii=False)
    print(json.dumps({"eligible": len(eligible), "sampled": len(chosen), "excluded_groups": len(seen)}))


def measure(db: Path, spec: Path, out: Path) -> None:
    sys.path.insert(0, str(HERE))
    import replay as R
    from living_memory.server import create_mcp_server
    from test_transport_identity import FakeMCP

    R.load_operator_env()
    data = json.loads(spec.read_text())
    mcp = create_mcp_server(db, mcp_factory=FakeMCP)
    lookup, recall = mcp.tools["memory_lookup"], mcp.tools["memory_recall"]
    obtained, chars, schemas = [], 0, 0
    for item in data["items"]:
        response = recall(item["query"], scope=item["scope"], max_results=10)
        chars += len(json.dumps(response, ensure_ascii=False))
        texts = []
        for result in response["results"]:
            node = result["node"]
            schemas += node.get("level") == "schema"
            text, cost = R.read(result, lookup)
            chars += cost
            texts.append(text)
        obtained.append(item["text"] in "\n".join(texts))
    delivered = []
    for query in data["collateral"]:
        response = recall(query["query"], scope=query["scope"], max_results=10)
        delivered.append([result["node"]["id"] for result in response["results"]])
    mcp.memory_store.close()
    json.dump({"obtained": obtained, "chars": chars, "schemas_delivered": schemas, "collateral": delivered}, out.open("w"))


def procedural_pass(db: Path) -> None:
    sys.path.insert(0, str(HERE))
    import replay as R
    from living_memory.storage import MemoryStore

    with MemoryStore(db) as store:
        print(json.dumps([R.procedural_pass(store), R.procedural_pass(store), R.census(store)["unconfirmed_steps"]]))


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    {"items": items, "measure": measure, "pass": procedural_pass}[mode](*map(Path, args))
