"""Full-text carrier regression runner; prints/saves aggregates only.

  python3 fulltext_run.py <code-src> prep <pristine.sqlite3> <out.sqlite3> <out.json>
  python3 fulltext_run.py <code-src> measure <db.sqlite3> <design|h2|h3> <cases.json|items3.json> <out.json>

<code-src> is imported before anything else and asserted, so the original
(24e9009) and the candidate run through the same reader in replay.py.
``prep`` copies the pristine snapshot, runs two ordinary procedural passes,
takes the census, and warms every searched scope with one query that matches
nothing, so recall's own drain embeds the rewritten carriers once; the
measured copies are taken from that state.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

CODE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(CODE))
import living_memory  # noqa: E402

assert Path(living_memory.__file__).resolve().is_relative_to(CODE), living_memory.__file__
HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE))
import holdout as H  # noqa: E402
import replay as R  # noqa: E402

assert Path(R.C.__file__).resolve().is_relative_to(CODE)
WARM = "zqxj vwkp qqzx"


def prep(pristine: Path, db: Path, out: Path, scopes: list[str]) -> None:
    from living_memory.server import create_mcp_server
    from living_memory.storage import MemoryStore
    from test_transport_identity import FakeMCP

    shutil.copyfile(pristine, db)
    res: dict = {"src": living_memory.__file__}
    with MemoryStore(db) as store:
        before = sum(len(n.content) for n in store.list_nodes(level="schema", include_decayed=False, limit=1_000_000)
                     if n.provenance.get("strategy") == "procedural")
        res["procedural_chars_before"] = before
        res["pass"] = [R.procedural_pass(store), R.procedural_pass(store)]
        census = R.census(store)
        census.pop("_unconfirmed"), census.pop("_visited")
        res["census"] = census
    started = time.perf_counter()
    mcp = create_mcp_server(db, mcp_factory=FakeMCP)
    for scope in scopes:
        mcp.tools["memory_recall"](WARM, scope=scope, max_results=1)
    mcp.memory_store.close()
    res["warm_seconds"] = round(time.perf_counter() - started, 1)
    json.dump(res, out.open("w"), indent=1)


def main() -> None:
    R.load_operator_env()
    mode = sys.argv[2]
    if mode == "prep":
        cases = json.load(open(os.environ["CASES"]))
        items = json.load(open(os.environ["ITEMS3"]))
        scopes = sorted({c[0] for n in cases["design"].values() for c in n} | {c["scope"] for c in cases["h2"]}
                        | {i["scope"] for i in items["items"]} | {q["scope"] for q in items["collateral"]})
        prep(Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]), scopes)
        return
    db, which, spec, out = Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5]), Path(sys.argv[6])
    if which == "h3":
        import holdout3

        holdout3.measure(db, spec, out)
        return
    cases = json.load(spec.open())
    res: dict = {"src": living_memory.__file__}
    if which == "design":
        for name in ("prose_recipes", "schema_topics"):
            res[name] = R.measure(db, [tuple(x) for x in cases["design"][name]], {}, {})
    else:
        res["h2"] = H.measure(db, cases["h2"], {})
    json.dump(res, out.open("w"))


if __name__ == "__main__":
    main()
