"""Explicit maintenance commands for Living Memory stores."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
import json
import sys

from living_memory.config import MemoryConfig, load_config
from living_memory.models import RetrievalPolicyFloors, RetrievalWeights
from living_memory.storage import MemoryStore

_WEIGHT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class RetrievalFloorChange:
    """Audited before/after reseat for one persisted retrieval-weight row."""

    scope: str
    family: str
    floor: RetrievalPolicyFloors | None
    before: RetrievalWeights
    after: RetrievalWeights

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "family": self.family,
            "floor": asdict(self.floor) if self.floor is not None else None,
            "before": _weights_to_dict(self.before),
            "after": _weights_to_dict(self.after),
        }


def reseat_retrieval_floors(
    store: MemoryStore,
    *,
    apply: bool = False,
) -> list[RetrievalFloorChange]:
    """Reseat persisted retrieval weights through the shared storage floor policy.

    Dry-run mode is the default: callers get the full audit list and no rows are
    written. Apply mode writes only rows whose before/after weights differ.
    """

    changes = retrieval_floor_changes(store)
    if apply:
        for change in changes:
            store.set_retrieval_weights(
                change.scope,
                bm25=change.after.bm25,
                vector=change.after.vector,
                graph=change.after.graph,
                learning_rate=change.after.learning_rate,
            )
    return changes


def retrieval_floor_changes(store: MemoryStore) -> list[RetrievalFloorChange]:
    """Return the audited floor reseats needed for persisted retrieval weights."""

    rows = store.connection.execute(
        """
        SELECT scope, bm25, vector, graph, learning_rate, updated_at
        FROM retrieval_weights
        ORDER BY scope
        """
    ).fetchall()
    changes: list[RetrievalFloorChange] = []
    for row in rows:
        before = RetrievalWeights(
            scope=str(row["scope"]),
            bm25=float(row["bm25"]),
            vector=float(row["vector"]),
            graph=float(row["graph"]),
            learning_rate=float(row["learning_rate"]),
            updated_at=str(row["updated_at"]),
        )
        after = store.apply_retrieval_weight_floors(before.scope, before)
        if not _weights_differ(before, after):
            continue
        family = _scope_family(before.scope)
        changes.append(
            RetrievalFloorChange(
                scope=before.scope,
                family=family,
                floor=store.config.retrieval_policy_floors.get(family),
                before=before,
                after=after,
            )
        )
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "reseat-floors":
        return _run_reseat_floors(args)
    parser.error("missing command")
    return 2


def _run_reseat_floors(args: Any) -> int:
    config = _resolve_config(db_path=args.db or args.sqlite_file, config_path=args.config)
    with MemoryStore(config) as store:
        changes = reseat_retrieval_floors(store, apply=args.apply)
    payload = {
        "command": "reseat-floors",
        "mode": "apply" if args.apply else "dry-run",
        "applied": bool(args.apply),
        "affected_count": len(changes),
        "changes": [change.to_dict() for change in changes],
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


def _resolve_config(
    *,
    db_path: str | Path | None,
    config_path: str | Path | None,
) -> MemoryConfig:
    config = load_config(config_path) if config_path is not None else MemoryConfig()
    if db_path is not None:
        config = replace(config, db_path=Path(db_path))
    return config


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run explicit Living Memory maintenance commands.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    reseat = subcommands.add_parser(
        "reseat-floors",
        help="Audit or apply retrieval-weight floor reseats for persisted rows.",
    )
    reseat.add_argument(
        "sqlite_file",
        nargs="?",
        help="SQLite database file. Defaults to the configured path.",
    )
    reseat.add_argument(
        "--db",
        dest="db",
        help="SQLite database file. Overrides the optional positional path.",
    )
    reseat.add_argument("--config", help="Optional TOML config path.")
    mode = reseat.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the reseat audit without writing rows. This is the default.",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Write the audited before/after reseats to the SQLite store.",
    )
    return parser


def _weights_differ(before: RetrievalWeights, after: RetrievalWeights) -> bool:
    return (
        abs(before.bm25 - after.bm25) > _WEIGHT_EPSILON
        or abs(before.vector - after.vector) > _WEIGHT_EPSILON
        or abs(before.graph - after.graph) > _WEIGHT_EPSILON
        or abs(before.learning_rate - after.learning_rate) > _WEIGHT_EPSILON
    )


def _weights_to_dict(weights: RetrievalWeights) -> dict[str, float]:
    return {
        "bm25": weights.bm25,
        "vector": weights.vector,
        "graph": weights.graph,
        "learning_rate": weights.learning_rate,
    }


def _scope_family(scope: str) -> str:
    if ":" in scope:
        return scope.split(":", 1)[0]
    return scope


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
