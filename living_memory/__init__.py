"""Checkout-local imports for the src-layout package."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_PACKAGE = _REPO_ROOT / "src" / "living_memory"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))

_DEPS_DIR = _REPO_ROOT / ".cache" / "python-deps"
if _DEPS_DIR.exists():
    _deps = str(_DEPS_DIR)
    if _deps not in sys.path:
        sys.path.insert(0, _deps)

from living_memory.config import MemoryConfig, RetrievalWeightConfig, load_config
from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity
from living_memory.feedback import (
    FeedbackService,
    ImplicitRecallFeedback,
    apply_pending_recall_feedback,
    apply_retrieval_feedback,
)
from living_memory.models import Connection, Node, PhaseInfo, RecallEvent, RetrievalWeights
from living_memory.phase import PhaseManager
from living_memory.retrieval import (
    MemoryRecallService,
    MemoryRetrievalService,
    RecallResult,
    memory_connect,
    memory_recall,
)
from living_memory.scope import ScopePlan, ScopeResolver, resolve_scope
from living_memory.storage import MemoryStore

__all__ = [
    "Connection",
    "FeedbackService",
    "ImplicitRecallFeedback",
    "LocalEmbeddingModel",
    "MemoryConfig",
    "MemoryRecallService",
    "MemoryRetrievalService",
    "MemoryStore",
    "Node",
    "PhaseInfo",
    "PhaseManager",
    "RecallEvent",
    "RecallResult",
    "RetrievalWeightConfig",
    "RetrievalWeights",
    "ScopePlan",
    "ScopeResolver",
    "apply_pending_recall_feedback",
    "apply_retrieval_feedback",
    "create_mcp_server",
    "cosine_similarity",
    "load_config",
    "memory_connect",
    "memory_recall",
    "resolve_scope",
    "run_server",
]


def __getattr__(name: str) -> object:
    if name in {"create_mcp_server", "run_server"}:
        from living_memory import server

        return getattr(server, name)
    raise AttributeError(name)
