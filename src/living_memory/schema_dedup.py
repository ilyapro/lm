"""Collapse same-procedure schema duplicates within one recall answer.

Stage of goal recall-precision. ``collapse_schema_duplicates`` takes the ranked
list (before the quality gate and the ``max_results`` cut) and returns
``(kept, collapsed)``: ``kept`` keeps the ranked order, ``collapsed`` holds the
lower-ranked duplicates it removed, so the next ranked results fill the freed
slots.

Behind ``LM_RECALL_SCHEMA_DEDUP`` (``1``/``true``/``yes``/``on``); unset or any
other value is the identity split.

"Same procedure" is the schema's title: the first line of its content,
``Procedure: <trigger>`` as ``consolidation._format_schema_content`` writes it,
whitespace-collapsed and casefolded. On the sfx store every such title maps to
one procedure_id, and the title reproduces the measured duplicate share; the
numbers are in docs/recall-schema-dedup.md. A schema with no trigger (bare
``Procedure``) or an empty first line has no title and is never collapsed.
Nodes other than schemas and group carriers are never collapsed.
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
    seen: set[str] = set()
    for result in ranked:
        title = schema_title(result.node)
        if title is None:
            kept.append(result)
        elif title in seen:
            collapsed.append(result)
        else:
            seen.add(title)
            kept.append(result)
    return kept, collapsed
