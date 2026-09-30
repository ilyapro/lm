"""Holdout 6 of holdout6-protocol.md: item selection; everything else is holdout4.py.

  python3 holdout6.py items <snapshot> <examined-copy> [<examined-copy> ...] <items.json>

Runs ``holdout4.items`` and then drops items this goal's own sessions wrote.
items.json holds private ids and texts: it stays in scratch, never in git.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import holdout4  # noqa: E402

OWN_TASK = "procedures-without-case-history"
OWN_PATTERN = "c32a8e82d028a68d"


def own(db: sqlite3.Connection, node_id: str) -> bool:
    row = db.execute("SELECT task, context FROM nodes WHERE id = ?", (node_id,)).fetchone()
    context = json.loads(row[1] or "{}") if row else {}
    task = " ".join(str(value) for value in (row[0] if row else "", context.get("task", "")))
    return OWN_TASK in task or context.get("task_pattern") == OWN_PATTERN


def items(snapshot: Path, *examined_and_out: Path) -> None:
    out = examined_and_out[-1]
    holdout4.items(snapshot, *examined_and_out)
    data = json.loads(out.read_text())
    dropped = 0
    with sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True) as db:
        for name, cases in data["families"].items():
            kept_cases = []
            for case in cases:
                kept = [item for item in case["items"] if not own(db, item["id"])]
                dropped += len(case["items"]) - len(kept)
                if kept:
                    kept_cases.append({**case, "items": kept})
            data["families"][name] = kept_cases
    data["overlap"]["own_goal_items"] = dropped
    out.write_text(json.dumps(data, ensure_ascii=False))
    counts = {
        name: [len(cases), sum(len(case["items"]) for case in cases),
               sum(item["instruction"] for case in cases for item in case["items"])]
        for name, cases in data["families"].items()
    }
    print(json.dumps({"overlap": data["overlap"], "cases_items_instructions": counts}))


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    {"items": items}[mode](*map(Path, args))
