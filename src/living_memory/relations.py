"""Correction semantics shared by runtime and offline relation readers.

Known duplicate bookkeeping is not a correction. Historical rows remain in
storage for inspection; they confer no validity, graph or lifecycle authority.
Unrecognized supersedes metadata retains its ordinary correction semantics.
"""

from collections.abc import Mapping
from typing import Any


_BOOKKEEPING_KINDS = ("drain_near_duplicate", "duplicate_content")
_HEURISTIC_PRODUCER = "lm_collapse_near_dups.py"
_HEURISTIC_REASON = "near-duplicate collapse"


def is_correction(relation_type: str, metadata: Mapping[str, Any]) -> bool:
    return relation_type == "supersedes" and not (
        metadata.get("kind") in _BOOKKEEPING_KINDS
        or (metadata.get("by") == _HEURISTIC_PRODUCER
            and metadata.get("reason") == _HEURISTIC_REASON)
    )


def correction_sql(alias: str = "") -> str:
    """SQL equivalent of is_correction; alias is a caller-owned identifier."""
    prefix = f"{alias}." if alias else ""
    kinds = ", ".join(f"'{kind}'" for kind in _BOOKKEEPING_KINDS)
    return (
        f"({prefix}type = 'supersedes'"
        f" AND COALESCE(json_extract({prefix}metadata, '$.kind'), '') NOT IN ({kinds})"
        f" AND NOT (COALESCE(json_extract({prefix}metadata, '$.by'), '') = '{_HEURISTIC_PRODUCER}'"
        f" AND COALESCE(json_extract({prefix}metadata, '$.reason'), '') = '{_HEURISTIC_REASON}'))"
    )


def semantic_relation_sql(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return f"({prefix}type <> 'supersedes' OR {correction_sql(alias)})"
