"""Collapse identical recorded procedure derivations within one recall answer.

The optional LM_RECALL_SCHEMA_DEDUP valve keeps the highest-ranked copy before
max_results, freeing its slot for the next result. Scope, group, recorded inputs
and content must all agree. Shared titles alone never hide independent knowledge;
unknown legacy derivations remain visible.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_DEDUP_ENV = "LM_RECALL_SCHEMA_DEDUP"
_ENABLED = frozenset({"1", "true", "yes", "on"})
_UNTITLED = frozenset({"", "procedure", "procedure:"})


def schema_dedup_enabled(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return str(source.get(SCHEMA_DEDUP_ENV) or "").strip().lower() in _ENABLED


def schema_title(node: Any) -> str | None:
    """Return the normalized title of a schema or group carrier, else ``None``."""

    level = getattr(node, "level", None)
    if level != "schema" and not (level == "concept" and node.context.get("trigger")):
        return None
    content = getattr(node, "content", None) or ""
    first_line = content.strip().split("\n", 1)[0]
    title = " ".join(first_line.split()).casefold()
    if title in _UNTITLED:
        return None
    return title


def collapse_schema_duplicates(
    ranked: Sequence[Any], *, env: Mapping[str, str] | None = None
) -> tuple[list[Any], list[Any]]:
    """Return ``(kept, collapsed)``; the identity split while the valve is off."""

    if not schema_dedup_enabled(env):
        return list(ranked), []
    kept: list[Any] = []
    collapsed: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for result in ranked:
        node = result.node
        identity = node.derivation_identity
        if node.materializer != "procedural" or identity is None:
            kept.append(result)
            continue
        key = (node.scope, node.level, identity, node.content)
        if key in seen:
            collapsed.append(result)
        else:
            seen.add(key)
            kept.append(result)
    return kept, collapsed
