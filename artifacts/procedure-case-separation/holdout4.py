"""Holdout 4 of holdout-protocol.md: the independent alt corpus; aggregates only.

  python3 holdout4.py items   <alt-snapshot> <examined-copy> [<another-examined-copy> ...] <items.json>
  LM_SRC=<checkout>/src python3 holdout4.py measure <db> <items.json> <alt-env> <out.json>
  python3 holdout4.py pass    <db> <out.json>       # two ordinary procedural passes + census
  python3 holdout4.py census  <db> <out.json>       # census only (the untouched copy)
  python3 holdout4.py report  <items.json> <before.json> <after.json> <before-census.json> <after-census.json>

``measure`` runs the code ``LM_SRC`` points at (imported before anything else,
and checked), so ``before`` is the original code on the untouched copy and
``after`` the candidate after its passes. items.json holds private ids and
texts: it stays in scratch, never in git.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("LM_SRC") or HERE.parents[1] / "src").resolve()
sys.path[:0] = [str(SRC), str(HERE.parents[1] / "tests")]

import living_memory  # noqa: E402  (first, so replay's own path cannot swap the code)

assert Path(living_memory.__file__).resolve().is_relative_to(SRC), living_memory.__file__
sys.path.insert(0, str(HERE))

import holdout as H  # noqa: E402
import replay as R  # noqa: E402
from living_memory import consolidation as C  # noqa: E402
from living_memory.storage import MemoryStore  # noqa: E402


def _digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def items(snapshot: Path, *examined_and_out: Path) -> None:
    *examined_copies, out = examined_and_out
    seen_ids, seen_texts = set(), set()
    for examined in examined_copies:
        with sqlite3.connect(f"file:{examined}?mode=ro", uri=True) as examined_db:
            for node_id, content in examined_db.execute("SELECT id, content FROM nodes"):
                seen_ids.add(node_id)
                seen_texts.add(_digest(content or ""))
    overlap = Counter()
    families: dict[str, list[dict]] = {"schema_topics": [], "own_name": []}
    with MemoryStore(snapshot) as store:
        edges = H.supersedes_sources(store)

        def unseen(node) -> bool:
            if node.id in seen_ids or _digest(node.content) in seen_texts:
                overlap["excluded_items"] += 1
                return False
            return True

        def current(node) -> bool:
            return node is not None and not node.decayed and not C._is_superseded(store, node.id)

        def instruction(node) -> bool:
            return bool(R.RECIPE_HEAD.match(node.content)) or bool(edges.get(node.id))

        # A: every live procedural schema with a readable trigger; its items are
        # the current records whose whole text the schema delivered.
        for schema in store.list_nodes(level="schema", include_decayed=False, limit=10**7):
            trigger = str(schema.context.get("trigger") or "")
            if schema.provenance.get("strategy") != "procedural" or not R.readable(trigger):
                continue
            case = []
            for node_id in schema.source_traces:
                node = store.get_node(node_id)
                if current(node) and node.content.strip() in schema.content and unseen(node):
                    case.append({"id": node.id, "text": node.content, "instruction": instruction(node)})
            if case:
                families["schema_topics"].append(
                    {"scope": schema.scope, "query": f"how to {trigger}", "items": case}
                )
        # B: every current instruction/correction record with a readable name,
        # queried by that name.
        for node in store.list_nodes(level="trace", include_decayed=False, limit=10**7):
            key = C._procedure_key(node)
            if key is None or not current(node) or not instruction(node):
                continue
            name = C._normalize_trigger(key.raw)
            if R.readable(name) and unseen(node):
                families["own_name"].append(
                    {
                        "scope": node.scope,
                        "query": f"how to {name}",
                        "items": [{"id": node.id, "text": node.content, "instruction": True}],
                    }
                )
    json.dump({"overlap": dict(overlap), "families": families}, out.open("w"), ensure_ascii=False)
    print(
        json.dumps(
            {
                "overlap": dict(overlap),
                **{name: [len(cases), sum(len(case["items"]) for case in cases)] for name, cases in families.items()},
            }
        )
    )


def load_env(path: Path) -> None:
    skip = {"LM_AUTH_TOKEN", "LM_DB_PATH", "LM_AUTO_CONSOLIDATE_POLICY"}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key and not key.startswith("#") and key.strip() not in skip:
            os.environ[key.strip()] = value.strip().strip('"')
    os.environ.pop("LM_AUTO_CONSOLIDATE_POLICY", None)


def measure(db: Path, spec: Path, env: Path, out: Path) -> None:
    from living_memory.server import create_mcp_server
    from test_transport_identity import FakeMCP

    load_env(env)
    data = json.loads(spec.read_text())
    mcp = create_mcp_server(db, mcp_factory=FakeMCP)
    lookup, recall = mcp.tools["memory_lookup"], mcp.tools["memory_recall"]
    result: dict[str, dict] = {}
    for name, cases in data["families"].items():
        obtained, chars, forced, schemas = [], 0, 0, []
        for case in cases:
            wanted = {item["id"] for item in case["items"]}
            response = recall(case["query"], scope=case["scope"], max_results=10)
            chars += len(json.dumps(response, ensure_ascii=False))
            texts = []
            for delivered in response["results"]:
                node = delivered["node"]
                if node.get("level") == "schema":
                    schemas.append(node["id"])
                text, cost = R.read(delivered, lookup, node["id"] in wanted)
                forced += cost
                texts.append(text)
            joined = "\n".join(texts)
            obtained.append([item["text"] in joined for item in case["items"]])
        result[name] = {"obtained": obtained, "chars": chars + forced, "forced": forced, "schemas": schemas}
    mcp.memory_store.close()
    json.dump({"src": str(SRC), **result}, out.open("w"))


def _census(db: Path, out: Path) -> dict:
    with MemoryStore(db) as store:
        census = R.census(store)
    census.pop("_visited")
    json.dump(census, out.open("w"))
    return census


def procedural_pass(db: Path, out: Path) -> None:
    with MemoryStore(db) as store:
        passes = [R.procedural_pass(store), R.procedural_pass(store)]
    census = _census(db, out)
    print(json.dumps({"passes": passes, "unconfirmed_steps": census["unconfirmed_steps"]}))


def census(db: Path, out: Path) -> None:
    print(json.dumps({"unconfirmed_steps": _census(db, out)["unconfirmed_steps"]}))


def report(spec: Path, before: Path, after: Path, census_before: Path, census_after: Path) -> None:
    data = json.loads(spec.read_text())
    was, now = json.loads(before.read_text()), json.loads(after.read_text())
    unconfirmed = {
        "before": json.loads(census_before.read_text()),
        "after": json.loads(census_after.read_text()),
    }
    out: dict = {"overlap": data["overlap"], "code": {"before": was["src"], "after": now["src"]}}
    for name, cases in data["families"].items():
        tally = Counter()
        for case, row_before, row_after in zip(cases, was[name]["obtained"], now[name]["obtained"]):
            for item, a, b in zip(case["items"], row_before, row_after):
                kind = "instruction" if item["instruction"] else "case"
                tally[f"{kind}_{'kept' if a and b else 'lost' if a else 'gained' if b else 'baseline_miss'}"] += 1
        out[name] = {
            "cases": len(cases),
            "items": dict(sorted(tally.items())),
            "H1_new_instruction_losses": tally["instruction_lost"],
            "chars_before_after": [was[name]["chars"], now[name]["chars"]],
            "forced_lookup_chars_before_after": [was[name]["forced"], now[name]["forced"]],
            "unconfirmed_steps_delivered_before_after": [
                sum(unconfirmed[side]["_unconfirmed"].get(node_id, 0) for node_id in run[name]["schemas"])
                for side, run in (("before", was), ("after", now))
            ],
        }
    out["H2_census_unconfirmed_before_after"] = [unconfirmed[side]["unconfirmed_steps"] for side in ("before", "after")]
    out["census_after"] = {k: v for k, v in unconfirmed["after"].items() if not k.startswith("_")}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    {"items": items, "measure": measure, "pass": procedural_pass, "census": census, "report": report}[mode](
        *map(Path, args)
    )
