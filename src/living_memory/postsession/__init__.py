"""Offline post-session extraction for Living Memory.

The stage reads a *finished* agent session's transcript and closes the write
half and the feedback loop of Living Memory without depending on the agent
having cooperated while it worked.

This package is the shared foundation every later child builds on:

``session``
    the stable schema -- :class:`~living_memory.postsession.session.SessionRecord`
    with its recall/write interactions, tool calls and file mutations, plus the
    :class:`~living_memory.postsession.session.ProposedOp` every downstream
    child emits. Published contract; see that module's docstring.
``judge``
    the pluggable, sandboxed LLM adapter the extractor and the correction
    detector both call.
``transcripts``
    one streaming adapter per recording format (claude, codex, ae_chat,
    ae_node_result, gigacode, deepseek).
``corpus``
    enumeration, hashing and the sealed train/eval/holdout split.
``evidence``
    the bounded, redacted, deterministic evidence list ``memory_attest``
    grades -- diff hunks and command outputs, never prose.

Only the schema is imported eagerly. ``transcripts``, ``corpus``, ``judge`` and
``evidence`` land in separate changes and drag in heavier dependencies, so they
are exposed through a module-level ``__getattr__``: ``postsession.load_transcript`` works
once ``transcripts.py`` exists, and importing the schema never fails because a
sibling module is absent or broken in an unrelated way.

Nothing in this package writes to Living Memory or reads its database.
"""

from __future__ import annotations

import importlib
from typing import Any

from .session import (
    CLIS,
    MEMORY_SERVER,
    OP_KINDS,
    RECALL_TOOL,
    REMEMBER_TOOL,
    SCHEMA_TYPES,
    SCHEMA_VERSION,
    SOURCES,
    TEACH_TOOL,
    DeliveredNode,
    Evidence,
    FileMutation,
    ProposedOp,
    ProposedOpError,
    Provenance,
    RecallInteraction,
    SessionRecord,
    SourceSpan,
    ToolCall,
    Turn,
    Wire,
    WriteInteraction,
    normalize_tool_name,
    published_fields,
)

#: Submodules whose public names are re-exported lazily, in lookup order.
_LAZY_SUBMODULES = ("transcripts", "corpus", "judge", "evidence")

__all__ = [
    "CLIS",
    "DeliveredNode",
    "Evidence",
    "FileMutation",
    "MEMORY_SERVER",
    "OP_KINDS",
    "ProposedOp",
    "ProposedOpError",
    "Provenance",
    "RECALL_TOOL",
    "REMEMBER_TOOL",
    "RecallInteraction",
    "SCHEMA_TYPES",
    "SCHEMA_VERSION",
    "SOURCES",
    "SessionRecord",
    "SourceSpan",
    "TEACH_TOOL",
    "ToolCall",
    "Turn",
    "Wire",
    "WriteInteraction",
    "normalize_tool_name",
    "published_fields",
]


def __getattr__(name: str) -> Any:
    """Resolve a name from a lazily-loaded submodule (see module docstring)."""

    if not name.startswith("_"):
        for module_name in _LAZY_SUBMODULES:
            qualified = f"{__name__}.{module_name}"
            try:
                module = importlib.import_module(qualified)
            except ModuleNotFoundError as exc:
                # Only swallow "this submodule does not exist yet"; a missing
                # dependency *inside* the submodule must still surface.
                if exc.name != qualified:
                    raise
                continue
            if hasattr(module, name):
                return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY_SUBMODULES))
