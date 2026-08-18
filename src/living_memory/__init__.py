"""Living Memory storage and recall services."""

from living_memory.config import MemoryConfig, RetrievalWeightConfig, load_config
from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity
from living_memory.feedback import (
    FeedbackService,
    ImplicitRecallFeedback,
    apply_pending_recall_feedback,
    apply_retrieval_feedback,
)
from living_memory.grounding import Grounding, containment, ground_results
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
    "Grounding",
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
    "containment",
    "cosine_similarity",
    "create_mcp_server",
    "ground_results",
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
