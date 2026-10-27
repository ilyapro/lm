"""FastMCP server surface for Living Memory."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import wraps
import inspect
from pathlib import Path
from threading import RLock, Timer
from types import CodeType, FunctionType, ModuleType
from typing import Annotated, Any, Literal
from uuid import uuid4
import hashlib
import json
import logging
import marshal
import os
import secrets
import sys
import time as _time

from pydantic import Field

from living_memory.auto_consolidation import (
    PENDING_KV_KEY as AUTO_CONSOLIDATION_PENDING_KV_KEY,
    AutoConsolidationScheduler,
)
from living_memory.chunking import TextChunk
from living_memory.config import MemoryConfig, load_config
from living_memory.consolidation import (
    _correction_text,
    DEFAULT_MIN_CLUSTER_SIZE,
    ConsolidationResult,
    ConsolidationService,
    TeachResult,
    prepare_teach,
)
from living_memory.decay import apply_decay
from living_memory.delivery import (
    DELIVERY_SNIPPET,
    DeliveryHistory,
    context_value_max_chars_from_env,
    full_node_diet_enabled_from_env,
    knowledge_node,
    provenance_value_max_chars_from_env,
    recall_delivery_records,
    session_dedup_enabled_from_env,
    shape_recall_results,
    snippet_ladder_from_env,
    snippet_max_chars_from_env,
    sparse_entries_enabled_from_env,
    stats_compaction_enabled_from_env,
)
from living_memory.edge_derivation import derive_edges_for_new_trace
from living_memory.models import Node, content_identity
from living_memory.prompts import MEMORY_GUIDANCE, retrieval_context_prompt
from living_memory.recall_map import RecallMapBuilder
from living_memory.resources import (
    connection_to_dict,
    global_concepts,
    memory_health as health_view,
    memory_stats,
    memory_status as status_view,
    node_to_dict,
    nodes_to_dict,
    project_concepts,
    recent_interactions,
)
from living_memory.retrieval import MemoryRecallService
from living_memory.scope import ScopePlan, normalize_scope, resolve_scope
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.feedback import (
    apply_explicit_marks,
    apply_lookup_credit,
    apply_pending_recall_feedback,
    has_explicit_marks,
    resolve_explicit_marks,
)
from living_memory import storage as storage_module
from living_memory.storage import MemoryStore
from living_memory.temporal import parse_timestamp
from living_memory.validity import correction_references

_BOOT_ID = uuid4().hex
_STARTED_AT = datetime.now(timezone.utc)
_RESTART_PENDING = False


def _loaded_code_identity(
    modules: dict[str, ModuleType] | None = None,
) -> dict[str, str | None]:
    """Snapshot LM function bytecode already held by this Python process.

    This deliberately does not infer a Git revision from mutable files or
    package metadata. It covers functions in modules loaded at this instant,
    including methods and functions behind ``functools.wraps`` decorators.
    """

    def normalized(code: CodeType) -> CodeType:
        return code.replace(
            co_filename="",
            co_consts=tuple(
                normalized(value) if isinstance(value, CodeType) else value
                for value in code.co_consts
            ),
        )

    entries: dict[str, bytes] = {}

    def add_function(key: str, function: FunctionType) -> None:
        seen: set[int] = set()
        while True:
            if key in entries or id(function) in seen:
                raise ValueError("ambiguous function identity")
            seen.add(id(function))
            # Marshal v1+ may encode a string differently depending on whether
            # Python interned it during import. Format 0 has no interned-string
            # tags and still serializes the complete normalized code object.
            entries[key] = marshal.dumps(normalized(function.__code__), 0)
            wrapped = getattr(function, "__wrapped__", None)
            if not isinstance(wrapped, FunctionType):
                break
            function = wrapped
            key += ":wrapped"

    identity: dict[str, str | None] = {
        "status": "unknown",
        "scheme": "python-loaded-functions-sha256-v2",
        "digest": None,
        "git_revision": None,
    }
    try:
        loaded = dict(modules if modules is not None else sys.modules)
        main_module = loaded.get("__main__")
        server_from_main = getattr(getattr(main_module, "__spec__", None), "name", None) == (
            "living_memory.server"
        )
        if server_from_main:
            if "living_memory.server" in loaded:
                raise ValueError("ambiguous server module")
            loaded["living_memory.server"] = main_module
        if "living_memory.server" not in loaded:
            raise ValueError("server module unavailable")
        for module_name, module in sorted(loaded.items()):
            if module_name != "living_memory" and not module_name.startswith(
                "living_memory."
            ):
                continue
            if not isinstance(module, ModuleType) or (
                module.__name__ != module_name
                and not (
                    server_from_main
                    and module_name == "living_memory.server"
                    and module is main_module
                    and module.__name__ == "__main__"
                )
            ):
                raise ValueError("ambiguous module identity")
            for name, value in sorted(vars(module).items()):
                if isinstance(value, FunctionType) and value.__module__ == module.__name__:
                    add_function(f"{module_name}.{name}", value)
                elif isinstance(value, type) and value.__module__ == module.__name__:
                    for member_name, member in sorted(vars(value).items()):
                        if isinstance(member, (staticmethod, classmethod)):
                            member = member.__func__
                        if isinstance(member, property):
                            for accessor_name in ("fget", "fset", "fdel"):
                                accessor = getattr(member, accessor_name)
                                if isinstance(accessor, FunctionType):
                                    add_function(
                                        f"{module_name}.{name}.{member_name}.{accessor_name}",
                                        accessor,
                                    )
                        elif isinstance(member, FunctionType):
                            add_function(f"{module_name}.{name}.{member_name}", member)
        if not entries:
            raise ValueError("no loaded LM functions")
        digest = hashlib.sha256()
        for name, code in sorted(entries.items()):
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(len(code).to_bytes(8, "big") + code)
        identity.update(status="known", digest=digest.hexdigest())
    except (AttributeError, TypeError, ValueError, OverflowError):
        pass
    return identity

# Server-wide KV key holding the persisted (rotated) bearer token. A value here
# takes precedence over the LM_AUTH_TOKEN env seed at startup, so a rotation via
# POST /admin/token survives a restart. The token lives only in the git-ignored
# SQLite state file — never logged, echoed in a response, or written to a tracked
# file, so it stays out of logs, diffs, and traces.
_AUTH_TOKEN_KV_KEY = "auth_token"

_AGENT_TOOL_NAMES = frozenset(
    {
        "memory_lookup",
        "memory_recall",
        "memory_remember",
        "memory_teach",
    }
)
_OPERATOR_TOOL_NAMES = frozenset(
    {
        "memory_connect",
        "memory_consolidate",
        "memory_forget",
        "memory_health",
        "memory_status",
    }
)

_TOOL_METRICS: dict[str, dict[str, Any]] = {}

def _track_latency(tool_name: str) -> Any:
    if tool_name not in _TOOL_METRICS:
        _TOOL_METRICS[tool_name] = {"count": 0, "latencies": deque(maxlen=1000)}

    def decorator(func: Any) -> Any:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = _time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = int((_time.perf_counter() - started) * 1000)
                _TOOL_METRICS[tool_name]["count"] += 1
                _TOOL_METRICS[tool_name]["latencies"].append(elapsed_ms)
        return wrapper
    return decorator


def create_mcp_server(
    db_path: str | Path | None = None,
    *,
    config: MemoryConfig | None = None,
    config_path: str | Path | None = None,
    default_scope: str | None = None,
    name: str = "Living Memory",
    mcp_factory: Any | None = None,
    auth_token: str | None = None,
    expose_attest: bool | None = None,
    expose_operator_tools: bool | None = None,
) -> Any:
    """Create a FastMCP server bound to one SQLite store.

    Maintenance handlers are always registered and directly callable. By
    default only the four agent tools are advertised by ``tools/list``;
    ``expose_operator_tools=True`` (or ``LM_EXPOSE_OPERATOR_TOOLS=1``) adds
    the five operator tools to discovery without changing call access.
    """

    store = MemoryStore(
        _resolve_config(
            db_path=db_path,
            config=config,
            config_path=config_path,
            default_scope=default_scope,
        )
    )
    # Pre-warm the embedding model so the first query does not pay the cold-start cost.
    LocalEmbeddingModel(model_name=store.config.embedding_model).warmup()
    mcp_cls = mcp_factory or _load_fastmcp()
    factory_kwargs: dict[str, Any] = {
        # Initialization has no current-work identity. Maps accompany recalls.
        "instructions": _server_instructions(store.config.default_scope),
    }
    # A token rotated earlier via POST /admin/token and persisted to the store
    # KV wins over the LM_AUTH_TOKEN env seed, so a rotation survives restart;
    # the env value only seeds the very first boot before any rotation.
    token_value = (store.get_kv(_AUTH_TOKEN_KV_KEY) or auth_token or "").strip()
    auth_provider = None
    # Only wire auth on the real FastMCP — custom test factories don't accept it.
    if token_value and mcp_factory is None:
        auth_provider = _build_static_token_auth(token_value)
        if auth_provider is not None:
            factory_kwargs["auth"] = auth_provider
    mcp = mcp_cls(name, **factory_kwargs)
    auth_state = _AuthTokenState(token_value, auth_provider)
    runtime_lock = RLock()
    _attach(mcp, "memory_store", store)
    _attach(mcp, "auth_token_state", auth_state)
    if expose_attest is None:
        expose_attest = os.environ.get("LM_EXPOSE_ATTEST", "") == "1"
    if expose_operator_tools is None:
        expose_operator_tools = (
            os.environ.get("LM_EXPOSE_OPERATOR_TOOLS", "") == "1"
        )
    _register_tools(mcp, store, runtime_lock, expose_attest=expose_attest)
    _configure_tool_list_visibility(
        mcp,
        expose_attest=expose_attest,
        expose_operator_tools=expose_operator_tools,
    )
    _register_resources(mcp, store, runtime_lock)
    _register_prompts(mcp, store, runtime_lock)
    _register_admin_routes(
        mcp,
        store=store,
        runtime_lock=runtime_lock,
        auth_state=auth_state,
        default_scope=store.config.default_scope,
        code_identity=_loaded_code_identity(),
    )
    return mcp


def _register_admin_routes(
    mcp: Any,
    *,
    store: MemoryStore,
    runtime_lock: Any,
    auth_state: "_AuthTokenState",
    default_scope: str,
    code_identity: dict[str, str | None],
) -> None:
    """Register /health, /admin/info, /admin/token, /admin/restart, /admin/decay-sweep."""

    register = getattr(mcp, "custom_route", None)
    if register is None:
        return
    try:
        from starlette.requests import Request
        from starlette.responses import JSONResponse
    except ImportError:
        return

    def _bearer_token(request: "Request") -> str:
        auth_header = request.headers.get("authorization", "")
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer":
            return ""
        return token.strip()

    def _unauthorized() -> "JSONResponse":
        return JSONResponse(
            {"ok": False, "error": "unauthorized"}, status_code=401
        )

    def _authorized(request: "Request") -> bool:
        expected = auth_state.token
        return bool(expected) and secrets.compare_digest(
            _bearer_token(request),
            expected,
        )

    @register("/health", methods=["GET"])
    async def health(request: "Request") -> "JSONResponse":
        if _RESTART_PENDING:
            # Body intentionally omits the substring "ok" so shell pollers that
            # gate on `grep -q ok` keep waiting until the post-exec process binds.
            return JSONResponse(
                {
                    "status": "restarting",
                    "service": "living-memory",
                    "boot_id": _BOOT_ID,
                    "code_identity": code_identity,
                },
                status_code=503,
            )
        return JSONResponse(
            {
                "ok": True,
                "service": "living-memory",
                "boot_id": _BOOT_ID,
                "code_identity": code_identity,
            },
            status_code=200,
        )

    @register("/admin/info", methods=["GET"])
    async def admin_info(request: "Request") -> "JSONResponse":
        if not _authorized(request):
            return _unauthorized()
        now = datetime.now(timezone.utc)
        return JSONResponse(
            {
                "ok": True,
                "service": "living-memory",
                "process_id": os.getpid(),
                "boot_id": _BOOT_ID,
                "code_identity": code_identity,
                "started_at": _STARTED_AT.isoformat(),
                "uptime_seconds": (now - _STARTED_AT).total_seconds(),
                "default_scope": default_scope,
                "argv": list(sys.argv),
            },
            status_code=200,
        )

    @register("/admin/token", methods=["POST"])
    async def admin_token(request: "Request") -> "JSONResponse":
        """Rotate the static bearer token (authorized by the *current* token).

        Persists the new token to the store KV (so the rotation survives a
        restart) and swaps it live on both the admin routes and the MCP
        StaticTokenVerifier, so the server immediately accepts only the new
        token. The token value is never echoed in the response.
        """

        if not _authorized(request):
            return _unauthorized()
        try:
            payload = await request.json()
        except Exception:
            payload = None
        new_token = ""
        if isinstance(payload, dict) and isinstance(payload.get("token"), str):
            new_token = payload["token"].strip()
        if not new_token:
            # Reject empty/missing tokens: an empty token would silently disable
            # bearer auth. The rejected value is never echoed back.
            return JSONResponse(
                {"ok": False, "error": "missing or empty token"},
                status_code=400,
            )
        rotated_at = datetime.now(timezone.utc).isoformat()
        with runtime_lock:
            # Persist first: if the process dies between persist and live-apply,
            # the durable value is still authoritative on the next startup.
            store.set_kv(_AUTH_TOKEN_KV_KEY, new_token)
            auth_state.rotate(new_token)
        return JSONResponse(
            {"ok": True, "rotated_at": rotated_at, "boot_id": _BOOT_ID},
            status_code=200,
        )

    @register("/admin/restart", methods=["POST"])
    async def admin_restart(request: "Request") -> "JSONResponse":
        if not _authorized(request):
            return _unauthorized()
        pid_before = os.getpid()
        restart_at = datetime.now(timezone.utc).isoformat()
        _schedule_self_exec()
        return JSONResponse(
            {
                "ok": True,
                "restart_at": restart_at,
                "pid_before": pid_before,
                "boot_id": _BOOT_ID,
            },
            status_code=202,
        )

    @register("/admin/decay-sweep", methods=["POST"])
    async def admin_decay_sweep(request: "Request") -> "JSONResponse":
        if not _authorized(request):
            return _unauthorized()
        started = _time.perf_counter()
        try:
            with runtime_lock:
                result = _decay_sweep_if_due(store, force=True)
        except Exception as exc:
            duration_ms = int((_time.perf_counter() - started) * 1000)
            return JSONResponse(
                {"ok": False, "error": str(exc), "duration_ms": duration_ms},
                status_code=500,
            )
        duration_ms = int((_time.perf_counter() - started) * 1000)
        payload: dict[str, Any] = {"ok": True, "duration_ms": duration_ms}
        if result is None:
            payload.update({"swept": False, "decayed_count": 0})
        else:
            payload.update(
                {
                    "swept": True,
                    "decayed_count": result["decayed_count"],
                    "expired_count": result["expired_count"],
                    "superseded_count": result["superseded_count"],
                    "last_decay_sweep_at": result["last_decay_sweep_at"],
                }
            )
        return JSONResponse(payload, status_code=200)


def _schedule_self_exec(delay_seconds: float = 0.2) -> None:
    """Fire os.execv after the current response has time to flush.

    `os.execv` replaces the current process image while preserving the PID;
    scheduling it on a daemon timer lets the 202 response complete first.
    Flipping `_RESTART_PENDING` makes the in-flight `/health` endpoint stop
    advertising readiness, so external pollers don't latch onto the dying
    process before exec replaces it.
    """

    global _RESTART_PENDING
    if _RESTART_PENDING:
        return
    _RESTART_PENDING = True

    def _exec_self() -> None:
        argv = [sys.executable, "-m", "living_memory.server", *sys.argv[1:]]
        os.execv(sys.executable, argv)

    timer = Timer(delay_seconds, _exec_self)
    timer.daemon = True
    timer.start()


def _token_entry(token: str) -> dict[str, dict[str, Any]]:
    """StaticTokenVerifier ``tokens`` mapping accepting exactly one bearer token."""

    return {token: {"client_id": "living-memory", "scopes": []}}


def _build_static_token_auth(token: str) -> Any | None:
    """Return a FastMCP static-token verifier or None if unsupported by factory."""

    try:
        from fastmcp.server.auth import StaticTokenVerifier
    except ImportError:
        return None
    return StaticTokenVerifier(tokens=_token_entry(token))


class _AuthTokenState:
    """Mutable holder for the server's single static bearer token.

    Shared by the admin-route authorizer and, when present, the FastMCP
    ``StaticTokenVerifier`` so a rotation via ``POST /admin/token`` takes effect
    live on both the admin routes and the MCP surface without a restart. The
    rotated value is persisted separately to the store KV so it also survives a
    restart. The token is never logged, echoed in a response, or written to a
    tracked file.
    """

    def __init__(self, token: str, verifier: Any | None = None) -> None:
        self._token = token
        self._verifier = verifier

    @property
    def token(self) -> str:
        return self._token

    def rotate(self, new_token: str) -> None:
        """Swap the accepted token on every live surface this holder feeds."""

        self._token = new_token
        verifier = self._verifier
        if verifier is not None:
            # StaticTokenVerifier.verify_token reads ``self.tokens.get(token)``
            # per request, so replacing the mapping rejects the old token and
            # accepts the new one on the MCP surface immediately.
            verifier.tokens = _token_entry(new_token)


_TRANSPORT_SESSION_KEY = "transport_session_id"


def _transport_session_id() -> str | None:
    """Derive the MCP transport session id for the current request, or None.

    FastMCP's ambient request context yields the client-echoed
    ``mcp-session-id`` header on streamable-HTTP and a stable per-connection
    UUID on stdio/SSE/in-memory transports. Outside a request context —
    direct tool-function calls, custom ``mcp_factory`` fakes — there is
    nothing to derive, so any failure degrades to None.
    """

    try:
        from fastmcp.server.dependencies import get_context

        session_id = get_context().session_id
    except Exception:
        return None
    if not session_id:
        return None
    return str(session_id)


def _with_transport_identity(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Stamp the derived transport session id into a context mapping.

    Returns the input unchanged when no transport identity is available;
    otherwise returns a copy (never mutates the caller's dict) with
    ``transport_session_id`` filled in only if the caller did not pass one —
    an explicit value always wins. The reserved key is deliberately NOT
    ``session_id``: scope resolution derives a ``session:<id>`` scope from
    that key, and the transport stamp must never perturb scope resolution.
    """

    session_id = _transport_session_id()
    if session_id is None:
        return data
    stamped = dict(data or {})
    stamped.setdefault(_TRANSPORT_SESSION_KEY, session_id)
    return stamped


def _server_instructions(default_scope: str, map_section: str = "") -> str:
    # Instructions carry the bootstrap, laws and knowledge/authority boundary.
    # Keep the data-derived map last so client clipping preserves that contract.
    # Tool descriptions also carry the boundary for clients omitting instructions.
    static = (
        "You and Living Memory form ONE cognitive system: you supply ephemeral "
        "reasoning, LM durable memory. Recall is perception — without it you "
        "act blind, inventing what memory already holds; remember is learning, "
        "teach is self-correction.\n\n"
        f"{MEMORY_GUIDANCE}\n\n"
        "## Three laws\n\n"
        "1. You MUST recall BEFORE you act and at every new turn of thought, "
        "including mid-work. When uncertain or stuck, recall.\n"
        "2. You MUST remember at the moment of insight, before your next "
        "action, but ONLY knowledge expensive to re-derive. Sessions end "
        "without warning; deferred learning is lost. One concrete fact per "
        "trace; never routine actions or copies of code.\n"
        "3. You MUST teach the moment a belief changes. Call `memory_teach` "
        "at once when recalled knowledge proves wrong: only teach creates "
        "the supersedes edge; a remember trace leaves stale knowledge active.\n\n"
        "## Tool protocol\n\n"
        "Follow the descriptions as protocol: `memory_recall` (BEFORE and "
        "MID-WORK triggers), `memory_remember` (write policy), `memory_teach` "
        "(corrections), `memory_lookup` (content_ref re-fetch). If this session "
        "defers tool schemas, you MUST load the Living Memory tools NOW and "
        "recall the task at hand.\n\n"
        "## Scope and context\n\n"
        "Write with a specific scope: `context.scope='project:<name>'` for "
        "project facts, `global` for universal knowledge. Read broad: omit "
        "`scope` to transfer across scopes. Set `task`, `agent`, and "
        "`session_id` in context to link work across sessions.\n"
        f"Your default scope is {default_scope}.\n"
    )
    section = map_section.strip()
    if not section:
        return static
    return f"{static}\n{section}\n"


# MCP-visible protocol must fit the per-tool client budget of 1024 chars.
# Both feedback arms share triggers and the knowledge/authority boundary.
_RECALL_DESCRIPTION = (
    "Recall before acting and at every new turn of thought. You MUST recall "
    "BEFORE acting: edits (conventions), state changes (concept before "
    "inventing), irreversible/outward steps (action plus target), world "
    "searches/measurements, design choices, new tasks/messages. You MUST "
    "recall MID-WORK for nearby knowledge; stuck or surprised means overdue, "
    "'if only I knew' means recall NOW. When uncertain, recall. Query "
    "identifiers; omit scope for cross-project knowledge. Depth 'causal' "
    "when debugging. "
    + MEMORY_GUIDANCE
    + " If delivered knowledge is insufficient, use content_ref via "
    "memory_lookup for full content."
    " below_threshold is an unqualified reference."
)

_REMEMBER_DESCRIPTION = (
    "memory_remember is ONLY for what is expensive to re-derive from "
    "code or git history: recipes, pitfalls, refutations, external "
    "contracts, measurement findings, a contract smeared across thousands "
    "of lines. The gate: cost, not possibility. You MUST remember at "
    "the moment of insight, before your next action — deferred remembers "
    "are usually lost; 'I wish I had known this earlier' means store it "
    "now. Triggers: non-obvious discovery, resolved failure and fix, "
    "a user correction or preference. One concrete fact per trace — "
    "paths, identifiers, the WHY; verified facts only — never routine "
    "actions, copies of code, speculation, unverified plans. An execution "
    "journal ('did X', 'node N done') re-derives trivially from git "
    "history — DO NOT store it: its noise drowns the rare real lesson "
    "in recall (done-journal-dump). A finished task earns "
    "ONE short closure note carrying only what the diff and git history "
    "cannot show (e.g. the invariant to preserve, the pitfall that cost "
    "time). Corrections NEVER go here — use memory_teach."
)

_ALTERNATIVES_CONSIDERED_DESCRIPTION = (
    "Rejected options behind this decision. Each item is an object "
    "{\"approach\": \"<what you did not do>\", \"rejected_because\": \"<why>\"}; "
    "both fields are non-empty strings."
)

_TEACH_DESCRIPTION = (
    "Store a corrective trace and connect it to the original via a "
    "supersedes edge. You MUST teach the moment a belief changes: a "
    "recalled fact proved wrong, or your understanding shifted and a prior "
    "trace reflects the old belief. Pass the original trace_id plus the "
    "corrected fact. Only memory_teach creates the supersedes edge — a "
    "memory_remember trace with UPDATE/CORRECTION wording does NOT count, "
    "and the stale node keeps outranking the correction in every future "
    "recall. Fixing a wrong assumption without teach (anti-pattern: "
    "silent-correction) leaves stale memory active: the system cannot "
    "self-correct silently."
)

# Feedback arms differ only in whether marks are requested or required.
EXPLICIT_FEEDBACK_PROMPT_ARMS: tuple[str, ...] = ("optional", "mandatory")
DEFAULT_EXPLICIT_FEEDBACK_PROMPT_ARM = "optional"

_RECALL_DESCRIPTION_MANDATORY = (
    _RECALL_DESCRIPTION
    + " You MUST mark each recall's results on your next call: used ids as "
    "used, off-topic ids as irrelevant."
)

_USED_FIELD_DESCRIPTION = (
    "Optional: ids of results from your earlier recalls this session that "
    "you used."
)
_IRRELEVANT_FIELD_DESCRIPTION = (
    "Optional: ids of earlier recall results that did not fit their query."
)
_USED_FIELD_DESCRIPTION_MANDATORY = (
    "REQUIRED after a recall: ids of the previous recall's results you used."
)
_IRRELEVANT_FIELD_DESCRIPTION_MANDATORY = (
    "REQUIRED after a recall: ids of the previous recall's results that were "
    "off-topic for its query."
)


#: Placeholder annotation of the ``used``/``irrelevant`` parameters. The real
#: one, carrying the active arm's field description, is installed by
#: ``_with_mark_fields``: ``from __future__ import annotations`` makes every
#: annotation a string that pydantic evaluates against module globals, so a
#: per-server description cannot be spelled inline.
_MarkIds = list[str] | None


def _with_mark_fields(texts: dict[str, str]) -> Any:
    """Decorator installing the arm's ``used``/``irrelevant`` field schemas."""

    def decorate(func: Any) -> Any:
        func.__annotations__["used"] = Annotated[
            list[str] | None, Field(description=texts["used"])
        ]
        func.__annotations__["irrelevant"] = Annotated[
            list[str] | None, Field(description=texts["irrelevant"])
        ]
        return func

    return decorate


def _explicit_feedback_prompt_arm() -> str:
    """Active description arm; an unknown value falls back to the default."""

    arm = os.environ.get("LM_EXPLICIT_FEEDBACK_PROMPT", "").strip().lower()
    if arm in EXPLICIT_FEEDBACK_PROMPT_ARMS:
        return arm
    return DEFAULT_EXPLICIT_FEEDBACK_PROMPT_ARM


def _explicit_feedback_texts(arm: str | None = None) -> dict[str, str]:
    """Tool and field descriptions for one arm (the active one by default)."""

    chosen = _explicit_feedback_prompt_arm() if arm is None else arm
    if chosen == "mandatory":
        return {
            "recall": _RECALL_DESCRIPTION_MANDATORY,
            "remember": _REMEMBER_DESCRIPTION,
            "teach": _TEACH_DESCRIPTION,
            "used": _USED_FIELD_DESCRIPTION_MANDATORY,
            "irrelevant": _IRRELEVANT_FIELD_DESCRIPTION_MANDATORY,
        }
    return {
        "recall": _RECALL_DESCRIPTION,
        "remember": _REMEMBER_DESCRIPTION,
        "teach": _TEACH_DESCRIPTION,
        "used": _USED_FIELD_DESCRIPTION,
        "irrelevant": _IRRELEVANT_FIELD_DESCRIPTION,
    }


def _effective_protocol_texts(default_scope: str, prompt_arm: str) -> dict[str, str]:
    """Static texts served for one explicitly selected prompt arm.

    The lookup docstring remains its single source. Inspecting the nested code
    object avoids constructing a store (and loading a model) just to check it.
    The data-derived Memory nearby section is deliberately outside this static
    projection; ``_server_instructions`` appends it at its canonical boundary.
    """
    if prompt_arm not in EXPLICIT_FEEDBACK_PROMPT_ARMS:
        raise ValueError(f"unknown explicit-feedback prompt arm: {prompt_arm}")
    lookup_description = None
    for constant in _register_tools.__code__.co_consts:
        if isinstance(constant, CodeType) and constant.co_name == "memory_lookup":
            raw = constant.co_consts[0] if constant.co_consts else None
            if isinstance(raw, str) and raw.strip():
                lookup_description = inspect.cleandoc(raw)
            break
    if not lookup_description:
        raise ValueError("memory_lookup has no usable docstring")
    feedback = _explicit_feedback_texts(prompt_arm)
    return {
        "server instructions": _server_instructions(default_scope),
        "memory_recall": feedback["recall"],
        "memory_remember": feedback["remember"],
        "memory_teach": feedback["teach"],
        "memory_lookup": lookup_description,
    }


_CONSOLIDATE_DESCRIPTION = (
    "Run one consolidation and decay maintenance pass. Consolidation "
    "promotes repeated know-how into level:schema procedural skills. "
    "Reusable know-how MUST take procedure form — trigger (when it fires), "
    "task_pattern (the recurring task class), procedure (the steps): set "
    "context.task_pattern and context.procedure_id on remember so "
    "promotion can match future recalls directly (the trigger_score "
    "channel). Store universal rules in global so they can become schemas "
    "— a universal rule stored as a project-local note gets re-invented "
    "elsewhere (anti-pattern: lookup-table-as-learning)."
)

def _tls_uvicorn_config(
    tls_cert: str | None, tls_key: str | None
) -> dict[str, Any] | None:
    """Translate an optional cert/key pair into uvicorn TLS kwargs.

    Returns None when neither is set (plain HTTP, unchanged behavior). Raises
    ValueError when exactly one is set, so a half-configured TLS never silently
    falls back to serving plaintext on the wire. FastMCP forwards these kwargs
    via ``run`` to ``run_http_async(uvicorn_config=...)`` to ``uvicorn.Config``.
    """

    cert = (tls_cert or "").strip()
    key = (tls_key or "").strip()
    if not cert and not key:
        return None
    if not cert or not key:
        raise ValueError(
            "TLS needs both a certificate and a key: set --tls-cert/LM_TLS_CERT "
            "and --tls-key/LM_TLS_KEY together (or neither for plain HTTP)."
        )
    return {"ssl_certfile": cert, "ssl_keyfile": key}


def _is_jsonrpc_notification(body: bytes) -> bool:
    """True when the body is a JSON-RPC notification (has ``method``, no ``id``).

    A batch counts only when every member is a notification. A request (carries
    ``id``) or an unparseable body returns False, so requests keep their normal
    handling — and their normal errors.
    """

    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return False
    if isinstance(data, list):
        return bool(data) and all(
            isinstance(m, dict) and "method" in m and "id" not in m for m in data
        )
    return isinstance(data, dict) and "method" in data and "id" not in data


class _SessionlessNotificationShim:
    """Answer 202 to a *session-less* JSON-RPC notification POST on ``/mcp``.

    Some MCP clients (notably Antigravity/``agy``) emit
    ``notifications/roots/list_changed`` before they have echoed back the
    ``Mcp-Session-Id`` that ``initialize`` returned. On a sub-millisecond loopback
    the id is already threaded; over a higher-latency path the notification leaves
    first, and the streamable-http session manager rejects a session-less message
    with 400 — which such clients treat as fatal and tear the whole connection
    down. The notification is fire-and-forget and Living Memory ignores client
    roots, so swallowing it with a 202 is lossless and leaves stateful sessions —
    and every request/response message — untouched. Requests (which carry an
    ``id``) are never intercepted, so genuine missing-session errors still surface.

    Implemented as pure ASGI (not ``BaseHTTPMiddleware``) so the SSE response
    stream is never buffered: only POST request bodies on the MCP path are
    inspected, and the response ``send`` channel is always passed straight through.
    """

    def __init__(self, app: Any, mount_path: str = "/mcp") -> None:
        self.app = app
        self._mount = mount_path.rstrip("/") or "/"

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        if (scope.get("path") or "").rstrip("/") != self._mount:
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v for k, v in scope.get("headers", [])}
        if headers.get("mcp-session-id"):  # already in a session -> leave alone
            await self.app(scope, receive, send)
            return
        # Buffer the request body so we can both inspect and (if needed) replay it.
        body = b""
        disconnected = False
        while True:
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
                if not message.get("more_body", False):
                    break
            else:  # http.disconnect or anything unexpected
                disconnected = True
                break
        if not disconnected and _is_jsonrpc_notification(body):
            from starlette.responses import Response

            await Response(status_code=202)(scope, receive, send)
            return
        replayed = False

        async def replay() -> dict:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            # Body already delivered: delegate to the real receive so a streaming
            # (SSE) response observes genuine client disconnects. Returning a
            # synthetic http.disconnect here aborts the reply mid-stream, which the
            # client sees as ``initialize: request terminated without response``.
            if disconnected:
                return {"type": "http.disconnect"}
            return await receive()

        await self.app(scope, replay, send)


def run_server(
    db_path: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    default_scope: str | None = None,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    tls_cert: str | None = None,
    tls_key: str | None = None,
) -> None:
    mcp = create_mcp_server(
        db_path=db_path,
        config_path=config_path,
        default_scope=default_scope,
        auth_token=os.environ.get("LM_AUTH_TOKEN"),
    )
    if transport == "stdio":
        mcp.run()
        return
    run_kwargs: dict[str, Any] = {"transport": transport, "host": host, "port": port}
    tls_config = _tls_uvicorn_config(tls_cert, tls_key)
    if tls_config is not None:
        run_kwargs["uvicorn_config"] = tls_config
    # Attach the session-less notification shim as ASGI middleware so a client that
    # fires notifications/roots/list_changed before it has threaded the session id
    # (e.g. agy over a higher-latency path) gets 202 instead of a fatal 400. Going
    # through FastMCP's run() (rather than serving uvicorn by hand) keeps its proper
    # signal/lifespan shutdown handling.
    try:
        from starlette.middleware import Middleware

        run_kwargs["middleware"] = [
            Middleware(_SessionlessNotificationShim, mount_path="/mcp")
        ]
    except ImportError:
        pass
    mcp.run(**run_kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    db_path = args.db or args.sqlite_file
    default_scope = args.default_scope or os.environ.get("LM_DEFAULT_SCOPE") or os.environ.get("LM_SCOPE")
    tls_cert = args.tls_cert or os.environ.get("LM_TLS_CERT")
    tls_key = args.tls_key or os.environ.get("LM_TLS_KEY")
    if args.embedding:
        os.environ["LIVING_MEMORY_EMBEDDING_BACKEND"] = args.embedding
    try:
        run_server(
            db_path=db_path,
            config_path=args.config,
            default_scope=default_scope,
            transport=args.transport,
            host=args.host,
            port=args.port,
            tls_cert=tls_cert,
            tls_key=tls_key,
        )
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _register_tools(
    mcp: Any, store: MemoryStore, runtime_lock: Any, *, expose_attest: bool = False
) -> None:
    recall_service = MemoryRecallService(store)
    recall_service._share_embedder_with_store()
    # One builder per server, like the service's own caches: the map's
    # stability guarantee is "same task, same structure across sessions",
    # which only holds if the thing remembering the structure outlives a call.
    recall_map_builder = RecallMapBuilder(store)
    consolidation_service = ConsolidationService(store)
    auto_consolidation = _build_auto_consolidation(store, runtime_lock, consolidation_service)
    _attach(mcp, "auto_consolidation", auto_consolidation)
    # Read once per server, like every description: the experiment switches
    # arms by restarting a sandbox server with LM_EXPLICIT_FEEDBACK_PROMPT.
    prompt_arm = _explicit_feedback_prompt_arm()
    feedback_texts = _explicit_feedback_texts(prompt_arm)
    protocol_texts = _effective_protocol_texts(store.config.default_scope, prompt_arm)
    mark_fields = _with_mark_fields(feedback_texts)

    @mcp.tool(description=protocol_texts["memory_remember"])
    @mark_fields
    @_track_latency("memory_remember")
    def memory_remember(
        content: str,
        context: dict[str, Any] | None = None,
        feedback: dict[str, Any] | None = None,
        alternatives_considered: Annotated[
            list[dict[str, Any]] | None,
            Field(description=_ALTERNATIVES_CONSIDERED_DESCRIPTION),
        ] = None,
        used: _MarkIds = None,
        irrelevant: _MarkIds = None,
    ) -> dict[str, Any]:
        """Store a new append-only trace.

        Returns a compact confirmation — node id/level/scope plus feedback
        counters, and a ``scheduled`` report when the write made a
        consolidation pass due (the pass runs in the background). Fetch full
        nodes via memory_lookup(node_id=...).

        The trace, alternatives and required recall feedback commit together.
        Current chunks are prepared before the store lock and applied inside
        the required transaction. A derived-index failure preserves the
        knowledge write and reports the affected ids as indexing deferred.
        Optional explicit marks run before implicit feedback and roll back
        with a failed required write; their own failures remain nonfatal.
        Edge derivation, instruction refresh and scheduling are nonfatal
        post-commit work. This does not make retries after response loss
        exactly-once.
        """

        alternatives = (
            storage_module._normalize_rejected_alternatives(alternatives_considered)
            if alternatives_considered is not None else None
        )
        contents = [content]
        contents.extend(
            storage_module._rejected_alternative_content(
                alternative["approach"], alternative["rejected_because"],
            ) for alternative in alternatives or ()
        )
        prepared = _prepare_trace_indices(store, contents)
        with runtime_lock:
            context = _with_transport_identity(context)
            try:
                auto_decay = _decay_sweep_if_due(store)
            except Exception:
                auto_decay = None
            # Knowledge and required recall feedback have one durable outcome.
            # Optional marks remain nonfatal, but roll back with a failed write.
            with store.transaction():
                rejected_nodes = []
                if alternatives is None:
                    node = store.append_trace(content, context, feedback=feedback)
                    rejected_alternatives: list[Any] | None = None
                else:
                    node, rejected_nodes = store.append_trace_with_rejected_alternatives(
                        content,
                        context,
                        feedback=feedback,
                        alternatives_considered=alternatives,
                    )
                    rejected_alternatives = [rejected.id for rejected in rejected_nodes]
                indexing = _index_written_traces(store, [node, *rejected_nodes], prepared)
                # Explicit marks before implicit feedback, so an irrelevant mark
                # carried on the closing call already keeps its edge out.
                marks = None
                if has_explicit_marks(used, irrelevant):
                    marks = apply_explicit_marks(
                        store,
                        used,
                        irrelevant,
                        via_tool="memory_remember",
                        source_id=node.id,
                        transport_session_id=_context_transport(context),
                        agent=_ambient_text(context, "agent"),
                    )
                implicit_feedback = apply_pending_recall_feedback(store, node)
                node = implicit_feedback.trace
            # Typed-edge derivation (edge_derivation.py go-rules) runs after
            # implicit recall feedback so an R6 reference upsert augments the
            # implicit related edge for the same pair instead of being
            # replaced by it. A derivation failure must never lose the write.
            try:
                derive_edges_for_new_trace(store, node)
            except Exception:
                pass
            scheduled = _schedule_auto_consolidation_if_due(
                store, auto_consolidation, node.scope,
                admitted_traces=1 + len(rejected_alternatives or []),
            )
            response = {
                "node": _remember_node_confirmation(node),
                "implicit_feedback": _implicit_feedback_to_dict(implicit_feedback),
                "auto_consolidation": scheduled,
                "auto_decay": auto_decay,
            }
            if rejected_alternatives is not None:
                response["rejected_alternatives"] = rejected_alternatives
            if marks is not None:
                response["feedback_marks"] = marks.summary()
            if indexing is not None:
                response["indexing"] = indexing
            return response

    @mcp.tool(description=protocol_texts["memory_teach"])
    @mark_fields
    @_track_latency("memory_teach")
    def memory_teach(
        trace_id: str,
        correction: str | dict[str, Any],
        confidence: float | None = None,
        context: dict[str, Any] | None = None,
        used: _MarkIds = None,
        irrelevant: _MarkIds = None,
    ) -> dict[str, Any]:
        """Commit the correction, provenance and accompanying marks together.

        Explicit marks precede implicit feedback and receive the corrective
        source before commit. Their own failures remain nonfatal. Optional
        consolidation scheduling runs only after the knowledge commit.
        Corrective trace chunks are prepared from normalized text before the
        store lock. Pure digest preparation also runs outside that lock; the
        writer checks its inputs against the current graph before using it.
        """

        correction = _correction_text(correction)
        prepared = _prepare_trace_indices(store, [correction])
        preparation = prepare_teach(
            store, trace_id, correction, confidence=confidence, guard=lambda: runtime_lock,
        )
        with runtime_lock:
            context = _with_transport_identity(context)
            with store.transaction():
                # Marks must precede implicit feedback, before the corrective
                # trace exists. A failed required write also rolls them back.
                marks = None
                if (
                    has_explicit_marks(used, irrelevant)
                    and store.get_node(trace_id) is not None
                    and _correction_text(correction)
                ):
                    marks = apply_explicit_marks(
                        store,
                        used,
                        irrelevant,
                        via_tool="memory_teach",
                        source_id="",
                        transport_session_id=_context_transport(context),
                        agent=_ambient_text(context, "agent"),
                    )
                taught = consolidation_service.memory_teach(
                    trace_id,
                    correction,
                    confidence=confidence,
                    context=context,
                    preparation=preparation,
                )
                indexing = _index_written_traces(store, [taught.corrective_trace], prepared)
                if marks is not None:
                    try:
                        store.set_feedback_marks_source(
                            marks.mark_ids, taught.corrective_trace.id
                        )
                    except Exception:
                        pass
            response = _teach_result_to_dict(taught)
            # The corrective trace counts toward the scope's cadence like any
            # write; the pass it makes due is scheduled, never run here.
            response["auto_consolidation"] = _schedule_auto_consolidation_if_due(
                store, auto_consolidation, taught.corrective_trace.scope
            )
            if marks is not None:
                response["feedback_marks"] = marks.summary()
            if indexing is not None:
                response["indexing"] = indexing
            return response

    @mcp.tool
    @_track_latency("memory_connect")
    def memory_connect(
        id_a: str,
        id_b: str,
        relation_type: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Connect two existing memory nodes."""

        with runtime_lock:
            connection = recall_service.memory_connect(
                id_a,
                id_b,
                relation_type,
                weight=weight,
                metadata=metadata,
            )
            return {"connection": connection_to_dict(connection)}

    @mcp.tool(description=protocol_texts["memory_recall"])
    @mark_fields
    @_track_latency("memory_recall")
    def memory_recall(
        query: str,
        scope: str | None = None,
        depth: int | str | None = 1,
        max_results: int = 5,
        ambient_context: dict[str, Any] | None = None,
        used: _MarkIds = None,
        irrelevant: _MarkIds = None,
        view: Annotated[
            Literal["knowledge", "full"],
            Field(description=(
                "knowledge returns flat content, identity, validity, delivery and "
                "content_ref. full returns diagnostic node wrappers, context and "
                "ranking scores with delivery compaction, plus recall-map "
                "selection accounting; memory_lookup by id "
                "with context_key or view='full' restores original metadata. "
                "Both views include superseded and scope-visible direct correction "
                "ids in superseded_by, even when corrections are not in results."
            )),
        ] = "knowledge",
    ) -> dict[str, Any]:
        """Retrieve relevant memory nodes using scope, text, vector, and graph signals.

        Each result carries a ``delivery`` class: ``full``, ``snippet`` (long
        content truncated inline), ``session_duplicate`` (already delivered on
        this transport session), ``twin_duplicate`` (byte-identical to a
        higher-ranked result with the same scope and source identity), or
        ``below_threshold`` (found source whose evidence did not qualify for
        inline content; its empty body and content_ref permit exact lookup).
        The evidence gate applies at every rank. In drop form, qualified
        results take priority and spare slots hold below-threshold references;
        stub form preserves the ranked cut. Both stay within max_results.
        Session dedup requires a transport session id and recorded delivery.
        Similarity ranks candidates without declaring them duplicates.
        The top-ranked content-bearer has a bounded default budget;
        lower-ranked bearers get descending budgets. Oversized content is
        selected for the query, while short content and complete schema recipes stay
        complete.
        The default ``view="knowledge"`` returns flat knowledge records with
        identity, content, correction/supersedes provenance and decay validity.
        Each includes ``delivery`` and a ``content_ref`` for complete content
        via ``memory_lookup``. ``view="full"`` returns diagnostic node wrappers
        with ranking scores and compacted context/provenance/stats. Original
        metadata is available through exact-id lookup's ``context_key`` or
        ``view="full"``. Both recall views use the same retrieval and delivery
        selection and keep all envelope fields, including warnings and maps.
        Map selection accounting (``sel``) is diagnostic: it remains in the
        stored map and ``view="full"``; knowledge keeps every other map field.
        ``superseded`` reports an incoming correction relation independently
        of correction availability; ``superseded_by`` lists scope-visible ids
        for exact lookup, including decayed history. Neither asserts that a
        correction is itself current. Chains and cycles retain direct links.

        When the ranked candidate pool holds more than the delivered results,
        an additive top-level ``recall_map`` describes the *residual*: a
        handful of labelled clusters with counts, a medoid example and a
        phrasing to ask with — what else memory holds for this task, as a plan
        for the next recall. It never reorders or replaces ``results``, and it
        is absent whenever there is no residual or nothing worth saying about
        it (``LM_RECALL_MAP=0`` removes it outright). Map construction and
        recording are optional: failures are logged and leave the selected
        knowledge intact. A map is returned only after its attribution is stored.

        Optional ``used``/``irrelevant`` mark results of *earlier* recalls on
        this transport session (docs/explicit-feedback.md); the response then
        carries a ``feedback_marks`` summary.
        """

        if view not in ("knowledge", "full"):
            raise ValueError("view must be 'knowledge' or 'full'")
        ambient_context = _with_transport_identity(ambient_context)
        raw_transport_id = (ambient_context or {}).get(_TRANSPORT_SESSION_KEY)
        # Mirror storage's stamping (str() of any non-None value) so the dedup
        # probe compares equal to the recall_events.transport_session_id column.
        transport_session_id = str(raw_transport_id) if raw_transport_id is not None else None
        session_dedup = bool(transport_session_id) and session_dedup_enabled_from_env()
        prepared_query = recall_service.prepare_query(query, max_results=max_results)
        with runtime_lock:
            auto_decay = _maybe_decay_sweep(store)
            # Resolve explicit marks before this call's own event exists, so a
            # mark can only name an earlier delivery; recorded after, with the
            # new event id as their source.
            pending_marks = (
                resolve_explicit_marks(store, used, irrelevant, transport_session_id)
                if has_explicit_marks(used, irrelevant)
                else None
            )
            # Read only prior formed responses. Selection and shaping share
            # this bounded observation; recording the current event adds no proof.
            try:
                delivery_history = DeliveryHistory(store.recall_delivery_history(
                    transport_session_id if session_dedup else None,
                ))
            except Exception:
                logging.getLogger(__name__).warning(
                    "could not read recall delivery history", exc_info=True,
                )
                delivery_history = DeliveryHistory()
            results = recall_service.memory_recall(
                prepared_query,
                scope=scope,
                ambient_context=ambient_context,
                depth=depth,
                max_results=max_results,
                select_results=lambda limited, residual, limit: _select_distinct_results(
                    limited, residual, limit, delivery_history
                ),
            )
            shaped = shape_recall_results(
                results,
                correction_refs=correction_references(
                    store,
                    {result.node.id: result.correction_ids for result in results},
                    resolve_scope(scope=scope, ambient_context=ambient_context, store=store),
                ),
                view=view,
                query=query,
                delivery_history=delivery_history,
                snippet_max_chars=snippet_max_chars_from_env(),
                context_value_max_chars=context_value_max_chars_from_env(),
                session_dedup=session_dedup,
                snippet_ladder=snippet_ladder_from_env(),
                full_node_diet=full_node_diet_enabled_from_env(),
                provenance_value_max_chars=provenance_value_max_chars_from_env(),
                stats_compaction=stats_compaction_enabled_from_env(),
                sparse_entries=sparse_entries_enabled_from_env(),
            )
            response: dict[str, Any] = {
                "query": query,
                "scope": scope,
                "recall_event_id": recall_service.last_recall_event_id,
                "count": len(shaped),
                "results": shaped,
                "auto_decay": auto_decay,
            }
            if pending_marks is not None:
                marks = apply_explicit_marks(
                    store,
                    used,
                    irrelevant,
                    via_tool="memory_recall",
                    source_id=recall_service.last_recall_event_id or "",
                    transport_session_id=transport_session_id,
                    agent=_ambient_text(ambient_context, "agent"),
                    resolved=pending_marks,
                )
                response["feedback_marks"] = marks.summary()
            # The required response is complete. Publish the optional map only
            # after its payload and attribution have been recorded together.
            residual = recall_service.last_residual
            if residual and _recall_map_enabled():
                try:
                    built = recall_map_builder.build(
                        residual,
                        relevance=recall_service.last_query_relevance,
                        scope=resolve_scope(
                            scope=scope, ambient_context=ambient_context, store=store,
                        ).requested_scope,
                        task=(ambient_context or {}).get("task"),
                        task_pattern=(ambient_context or {}).get("task_pattern"),
                        transport_session_id=transport_session_id,
                    )
                    if built is not None and recall_service.last_recall_event_id is not None:
                        payload = built.to_dict()
                        store.attach_recall_map(
                            recall_service.last_recall_event_id,
                            payload,
                        )
                        response["recall_map"] = (
                            payload if view == "full"
                            else {key: value for key, value in payload.items() if key != "sel"}
                        )
                except Exception:
                    logging.getLogger(__name__).warning(
                        "could not deliver optional recall map for event %s",
                        recall_service.last_recall_event_id,
                        exc_info=True,
                    )
            if recall_service.last_recall_event_id is not None:
                try:
                    store.record_recall_delivery(
                        recall_service.last_recall_event_id,
                        recall_delivery_records(results, shaped),
                    )
                except Exception:
                    logging.getLogger(__name__).warning(
                        "could not record recall delivery for event %s",
                        recall_service.last_recall_event_id, exc_info=True,
                    )
        # Finish this call's event, feedback, map and delivery while last_* still
        # belongs to it. Optional repair may wait without holding up store reads.
        recall_service.refresh_legacy_passages([
            result for result, entry in zip(results, shaped, strict=True)
            if entry["delivery"] == DELIVERY_SNIPPET
        ], guard=lambda: runtime_lock)
        return response

    # memory_attest is NOT registered by default (expose_attest=False). Field
    # data (2026-08-20, both hosts): agents folded it into their closure
    # ritual, attesting their own same-session recalls — the case
    # memory_remember already closes implicitly — with prose evidence that
    # fails containment (credited 0/21 and 1/6). Every session paid ~250
    # context tokens for a tool that never earned credit. The machinery
    # stays: attestation.py, the recall_attestations table, and this guarded
    # registration serve the post-session extraction runner — enable via
    # LM_EXPOSE_ATTEST=1 (or expose_attest=True) when that stage goes live
    # (contract: docs/post-session-attestation.md).
    if expose_attest:
        from living_memory.attestation import attest_recall_usage

        @mcp.tool
        @_track_latency("memory_attest")
        def memory_attest(
            recall_event_id: str,
            evidence: list[str],
            context: dict[str, Any] | None = None,
            trace_id: str | None = None,
        ) -> dict[str, Any]:
            """Offline attestation of a FINISHED session's recall — extraction
            tooling, not an agent action. Recalls of the current session are
            closed by your memory_remember automatically; do NOT attest them.
            Full contract: docs/post-session-attestation.md."""

            context = _with_transport_identity(context)
            with runtime_lock:
                return attest_recall_usage(
                    store,
                    recall_event_id,
                    evidence,
                    context=context,
                    trace_id=trace_id,
                )


    @mcp.tool(description=protocol_texts["memory_lookup"])
    @_track_latency("memory_lookup")
    def memory_lookup(
        scope: str | None = None,
        task_pattern: str | None = None,
        procedure_id: str | None = None,
        lesson_kind: str | None = None,
        level: str | None = "trace",
        node_id: str | None = None,
        node_ids: list[str] | None = None,
        view: str = "knowledge",
        context_key: str | None = None,
    ) -> dict[str, Any]:
        """Return memory nodes by exact context-field match, or knowledge by id.

        ``node_id``/``node_ids`` fetch full content, identity, and validity
        without unrelated context or provenance history. Use ``context_key``
        to include one original context field, or ``view="full"`` for the
        complete stored node. These selectors apply only to id fetches and
        cannot be combined. Missing ids are skipped and
        listed under ``missing``; decayed nodes are returned with their
        ``decayed`` flag set; ``level`` is ignored for id fetches. A ``scope``
        passed alongside ids verifies instead of filtering: mismatching nodes
        stay in ``results`` and are reported in ``scope_mismatches``.
        ``superseded`` reports known replacement even without a readable
        correction; ``superseded_by`` lists direct correction ids allowed by
        scope (including decayed history). These fields accompany both views.
        """

        with runtime_lock:
            normalized_scope = normalize_scope(scope) if scope is not None else None
            filters = {
                key: value
                for key, value in {
                    "task_pattern": task_pattern,
                    "procedure_id": procedure_id,
                    "lesson_kind": lesson_kind,
                }.items()
                if value is not None
            }
            requested_ids: list[str] = []
            for candidate in [node_id, *(node_ids or [])]:
                if candidate and candidate not in requested_ids:
                    requested_ids.append(candidate)

            if requested_ids:
                if filters:
                    return {
                        "error": "node_id/node_ids cannot be combined with context filters",
                        "scope": normalized_scope,
                        "filters": dict(filters),
                        "count": 0,
                        "results": [],
                    }
                if view not in ("knowledge", "full") or (
                    context_key is not None and (not context_key or view == "full")
                ):
                    return {
                        "error": (
                            "use view='knowledge' or view='full'; context_key "
                            "requires view='knowledge' and a nonempty key"
                        ),
                        "scope": normalized_scope,
                        "filters": {},
                        "count": 0,
                        "results": [],
                    }
                found = store.get_nodes(requested_ids)
                results = []
                for node in nodes_to_dict(
                    store, [found[requested] for requested in requested_ids if requested in found],
                    resolve_scope(scope=normalized_scope, store=store),
                ):
                    if view == "full":
                        results.append(node)
                        continue
                    knowledge = knowledge_node(node)
                    knowledge.update({key: node[key] for key in ("superseded", "superseded_by")})
                    if context_key is not None:
                        knowledge["context"] = (
                            {context_key: node["context"][context_key]}
                            if context_key in node["context"]
                            else {}
                        )
                    results.append(knowledge)
                response: dict[str, Any] = {
                    "scope": normalized_scope,
                    "filters": {},
                    "count": len(results),
                    "results": results,
                    "missing": [
                        requested for requested in requested_ids if requested not in found
                    ],
                }
                if normalized_scope is not None:
                    response["scope_mismatches"] = [
                        {"id": entry["id"], "scope": entry["scope"]}
                        for entry in results
                        if entry["scope"] != normalized_scope
                    ]
                # The id-fetch path, and only it: naming an exact ULID is an
                # act somebody had to be handed the id for, which is what
                # makes it the one honest "I followed the card" signal the
                # server can see. The context-filter path below is a query,
                # not a follow, and records nothing.
                #
                # This stays a read of the *nodes*: the event lands in its own
                # table and no access counter, recall event, or pending
                # feedback moves. Ids are recorded as requested, missing ones
                # included, because the request is the signal.
                #
                # Then the signal is spent: a requested id that a recall on
                # this same transport delivered within the window earns the
                # usage credit a grounded closure would have given it
                # (feedback.apply_lookup_credit — usefulness, the event
                # scope's weights, one anchor edge), once per (event, node)
                # across both signals via the recall credit ledger.
                # LM_LOOKUP_CREDIT_POLICY=off keeps the record and skips the
                # credit. The instant is read once, from the storage clock
                # that stamps every other ledger row (recall events, delivery
                # history), so the event row, the delivery-window correlation
                # and the credit join all see the same lookup time — a wall
                # clock read here would drift from a frozen or offset storage
                # clock and put the lookup outside its own delivery window.
                # Usage observation is optional: a recording failure leaves
                # the assembled read response intact and skips credit. Its
                # mutation outcome may be unknown, so it is never retried here.
                # A failed credit is logged inside apply_lookup_credit.
                identity = _with_transport_identity(None) or {}
                transport_session_id = identity.get(_TRANSPORT_SESSION_KEY)
                occurred_at = storage_module._utc_now()
                try:
                    lookup_event_id = store.record_lookup_event(
                        requested_ids,
                        transport_session_id=transport_session_id,
                        occurred_at=occurred_at,
                    )
                except Exception as error:
                    logging.getLogger(__name__).warning(
                        "lookup usage recording failed (%s; %d requested id(s)); credit skipped",
                        type(error).__name__[:80],
                        len(requested_ids),
                    )
                    return response
                if lookup_event_id is not None:
                    apply_lookup_credit(
                        store,
                        lookup_event_id,
                        requested_ids,
                        transport_session_id,
                        occurred_at,
                    )
                return response

            if view != "knowledge" or context_key is not None:
                return {
                    "error": "view and context_key require node_id or node_ids",
                    "scope": normalized_scope,
                    "filters": dict(filters),
                    "count": 0,
                    "results": [],
                }
            if not filters:
                return {
                    "error": "at least one exact context filter is required",
                    "scope": normalized_scope,
                    "filters": {},
                    "count": 0,
                    "results": [],
                }
            if normalized_scope is None:
                return {
                    "error": "scope is required with context filters",
                    "scope": None,
                    "filters": dict(filters),
                    "count": 0,
                    "results": [],
                }
            nodes = store.list_nodes_by_context(
                scope=normalized_scope,
                context_filters=filters,
                level=level,  # type: ignore[arg-type]
            )
            return {
                "scope": normalized_scope,
                "filters": dict(filters),
                "count": len(nodes),
                "results": nodes_to_dict(store, nodes, ScopePlan(normalized_scope, (normalized_scope,))),
            }

    @mcp.tool(description=_CONSOLIDATE_DESCRIPTION)
    @_track_latency("memory_consolidate")
    def memory_consolidate(scope: str | None = None, force: bool = False) -> dict[str, Any]:
        """Run one consolidation and decay maintenance pass."""

        with runtime_lock:
            auto_decay = _maybe_decay_sweep(store)
            result = consolidation_service.memory_consolidate(scope=scope, force=force)
            payload = _consolidation_result_to_dict(result)
            payload["auto_decay"] = auto_decay
            return payload

    @mcp.tool
    @_track_latency("memory_forget")
    def memory_forget(id: str, reason: str | None = None) -> dict[str, Any]:
        """Soft-delete a memory node while preserving the stored record."""

        with runtime_lock:
            node = consolidation_service.memory_forget(id, reason)
            return {"node": node_to_dict(node)}

    @mcp.tool
    @_track_latency("memory_status")
    def memory_status(scope: str | None = None) -> dict[str, Any]:
        """Describe current memory phase, coverage, confidence, and policy."""

        with runtime_lock:
            return status_view(store, scope=scope)

    @mcp.tool
    @_track_latency("memory_health")
    def memory_health(
        scope: str | None = None,
        window_hours: int = 168,
        top_stale: int = 5,
    ) -> dict[str, Any]:
        """Report activity ratios, dedup density, staleness, and retrieval policy."""

        with runtime_lock:
            return health_view(
                store,
                scope=scope,
                window_hours=window_hours,
                top_stale=top_stale,
            )


def _register_resources(mcp: Any, store: MemoryStore, runtime_lock: Any) -> None:
    @mcp.resource("memory://global/concepts")
    def resource_global_concepts() -> dict[str, Any]:
        with runtime_lock:
            return global_concepts(store)

    @mcp.resource("memory://project/{name}/concepts")
    def resource_project_concepts(name: str) -> dict[str, Any]:
        with runtime_lock:
            return project_concepts(store, name)

    @mcp.resource("memory://stats")
    def resource_stats() -> dict[str, Any]:
        with runtime_lock:
            return memory_stats(store)

    @mcp.resource("memory://recent")
    def resource_recent() -> dict[str, Any]:
        with runtime_lock:
            return recent_interactions(store)

    @mcp.resource("memory://latency")
    def resource_latency() -> dict[str, Any]:
        metrics = {}
        for tool_name, data in _TOOL_METRICS.items():
            count = data["count"]
            if count == 0:
                continue
            latencies = sorted(data["latencies"])
            n = len(latencies)
            metrics[tool_name] = {
                "count": count,
                "p50_ms": latencies[int(n * 0.50)],
                "p95_ms": latencies[int(n * 0.95)],
                "p99_ms": latencies[int(n * 0.99)],
            }
        return {"uri": "memory://latency", "metrics": metrics}


def _register_prompts(mcp: Any, store: MemoryStore, runtime_lock: Any) -> None:
    @mcp.prompt(name="memory://prompt/retrieval_context")
    def retrieval_context(
        task: str = "",
        scope: str | None = None,
        max_concepts: int = 5,
        min_confidence: float = 0.0,
        retrieval_policy: str = "balanced",
        agent: str | None = None,
        ambient_context: dict[str, Any] | None = None,
    ) -> str:
        """Return a formatted active memory context block."""

        with runtime_lock:
            return retrieval_context_prompt(
                store,
                task=task,
                scope=scope,
                max_concepts=max_concepts,
                min_confidence=min_confidence,
                retrieval_policy=retrieval_policy,
                agent=agent,
                ambient_context=ambient_context,
            )


_ADAPTIVE_MIN_WRITES = 5


def _adaptive_trigger_step(write_count: int) -> int | None:
    """Trigger step for adaptive auto-consolidate policy.

    Counts admitted traces, including replacements and decayed traces:
    every 5 writes up to 50, every 25 up to 500, then the fixed cadence.
    """

    if write_count < _ADAPTIVE_MIN_WRITES:
        return None
    if write_count < 50:
        return 5
    if write_count < 500:
        return 25
    return DEFAULT_MIN_CLUSTER_SIZE


def _auto_consolidate_policy() -> str:
    return os.environ.get("LM_AUTO_CONSOLIDATE_POLICY", "fixed").strip().lower()


def _recall_map_enabled() -> bool:
    """Read ``LM_RECALL_MAP``: default on, ``"0"`` removes the key.

    The rollback valve for the additive ``recall_map``, in the shape of the
    ``LM_DELIVERY_*`` valves: on unless explicitly switched off, and switching
    it off restores a byte-identical pre-map response rather than an empty map.
    """

    return os.environ.get("LM_RECALL_MAP", "").strip() != "0"


def _select_distinct_results(
    limited: list[Any],
    residual: list[Any],
    max_results: int,
    delivery_history: DeliveryHistory,
) -> tuple[list[Any], list[Any]]:
    """Spend a bounded cut on distinct knowledge when a repeat hides a fact.

    Keep the ordinary cut unless a classified repeat can actually be replaced
    by a later independent result. Session repeats remain selectable: a caller
    may intentionally ask again for the prior fact. Only exact content with
    matching scope and source identity can release a slot.
    """

    if len(limited) < max_results or not residual:
        return limited, residual
    pool = limited + residual[:max_results]
    seen_content: dict[tuple[Any, ...], str] = {}
    selected: list[Any] = []
    for result in pool:
        node = result.node
        identity = content_identity(node) if node.content else None
        bearer = seen_content.get(identity) if identity is not None else None
        if identity is not None and bearer is None:
            seen_content[identity] = node.id
        if (
            bearer in {item.node.id for item in selected}
            and not delivery_history.has_content(node)
        ):
            continue
        selected.append(result)
        if len(selected) == max_results:
            break
    if len(selected) < max_results:
        return limited, residual
    selected_ids = {result.node.id for result in selected}
    if selected_ids == {result.node.id for result in limited}:
        return limited, residual
    return selected, [result for result in limited + residual if result.node.id not in selected_ids]


def _context_transport(context: dict[str, Any] | None) -> str | None:
    """The transport id a write is stamped with, as storage stamps it."""

    value = (context or {}).get(_TRANSPORT_SESSION_KEY)
    return str(value) if value is not None and str(value) else None


def _ambient_text(ambient_context: dict[str, Any] | None, key: str) -> str | None:
    """One ambient field as a non-empty string, or None.

    ``str()`` of whatever was passed, the same coercion storage stamps its
    ``task`` column with, so the map's cache key is built from the value the
    recall event records — plus: blank is absent. A whitespace-only ``task``
    would otherwise key a cache entry that no later recall can hit and that
    every task-less recall would collide on.
    """

    if not ambient_context:
        return None
    value = ambient_context.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_DECAY_SWEEP_KV_KEY = "last_decay_sweep_at"
_DECAY_SWEEP_DEFAULT_INTERVAL_SEC = 3600


def _decay_sweep_interval_seconds() -> int:
    """Read LM_DECAY_SWEEP_INTERVAL_SEC. 0 disables time-based sweep."""

    raw = os.environ.get("LM_DECAY_SWEEP_INTERVAL_SEC")
    if raw is None or not raw.strip():
        return _DECAY_SWEEP_DEFAULT_INTERVAL_SEC
    try:
        value = int(raw.strip())
    except ValueError:
        return _DECAY_SWEEP_DEFAULT_INTERVAL_SEC
    return max(0, value)


def _decay_sweep_if_due(
    store: MemoryStore,
    *,
    force: bool = False,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Run a scope-agnostic decay sweep when the time gate has elapsed.

    Returns a summary dict when a sweep ran, or None when it was skipped
    (interval disabled, or last sweep too recent). Raises on storage or
    decay errors — callers decide whether to propagate or swallow.
    """

    interval = _decay_sweep_interval_seconds()
    if not force and interval <= 0:
        return None

    current = now or datetime.now(timezone.utc)

    if not force:
        last_raw = store.get_kv(_DECAY_SWEEP_KV_KEY)
        if last_raw:
            last = parse_timestamp(last_raw)
            if last is not None and (current - last).total_seconds() < interval:
                return None

    started = _time.perf_counter()
    result = apply_decay(store, scope=None, now=current)
    duration_ms = int((_time.perf_counter() - started) * 1000)

    timestamp_iso = current.astimezone(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    store.set_kv(_DECAY_SWEEP_KV_KEY, timestamp_iso)

    return {
        "swept": True,
        "decayed_count": len(result.expired) + len(result.superseded),
        "expired_count": len(result.expired),
        "superseded_count": len(result.superseded),
        "duration_ms": duration_ms,
        "last_decay_sweep_at": timestamp_iso,
    }


def _maybe_decay_sweep(store: MemoryStore) -> dict[str, Any] | None:
    """Opportunistic time-based sweep that never raises into the caller."""

    try:
        return _decay_sweep_if_due(store)
    except Exception:
        return None


_AUTO_CONSOLIDATION_STEP_HANDOFF_SECONDS = 0.005


def _auto_consolidation_due(
    store: MemoryStore, scope: str, admitted_traces: int = 1,
) -> int | None:
    """Active population when the committed writes crossed an interval.

    Trace rows retain admitted knowledge, including replacements and soft
    deletions. Their scoped covering index supplies the cadence without a
    separate progress record. A remember with alternatives can cross a
    boundary without ending exactly on it.
    """

    row = store.connection.execute(
        "SELECT COUNT(*) AS count FROM nodes WHERE level = 'trace' AND scope = ?",
        (scope,),
    ).fetchone()
    write_count = int(row["count"])
    step = (
        _adaptive_trigger_step(write_count)
        if _auto_consolidate_policy() == "adaptive"
        else DEFAULT_MIN_CLUSTER_SIZE
    )
    if step is None or write_count < step:
        return None
    if write_count // step == max(0, write_count - admitted_traces) // step:
        return None

    # Population supplies material, not cadence. Consolidation owns the
    # merge floors and sizes them from the active traces it actually loads.
    row = store.connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM nodes
        WHERE level = 'trace' AND scope = ? AND decayed = 0
        """,
        (scope,),
    ).fetchone()
    return int(row["count"]) or None


def _auto_consolidate_if_due(
    store: MemoryStore,
    consolidation_service: ConsolidationService,
    scope: str,
) -> dict[str, Any] | None:
    """Run the due pass inline and return its summary (no scheduler)."""

    if _auto_consolidation_due(store, scope) is None:
        return None
    return _compact_consolidation_summary(
        consolidation_service.memory_consolidate(scope=scope, force=False)
    )


def _schedule_auto_consolidation_if_due(
    store: MemoryStore,
    scheduler: AutoConsolidationScheduler,
    scope: str,
    *,
    admitted_traces: int = 1,
) -> dict[str, Any] | None:
    """Schedule the pass this write made due; report, never run it.

    The write path answers at write time: the report says the pass is
    ``scheduled`` (``coalesced`` when it joins one already queued or
    running), and the pass's own summary lands in the scheduler's
    ``last_result(scope)`` when the worker finishes it.

    Both write tools call this after their required transaction commits.
    Optional scheduling failures are logged and cannot replace that outcome.
    """

    try:
        trace_count = _auto_consolidation_due(store, scope, admitted_traces)
        if trace_count is None:
            return None
        return scheduler.request(scope, trace_count=trace_count)
    except Exception:
        logging.getLogger(__name__).warning(
            "could not schedule post-commit auto-consolidation for %s", scope, exc_info=True
        )
        return None


def _build_auto_consolidation(
    store: MemoryStore,
    runtime_lock: Any,
    consolidation_service: ConsolidationService,
) -> AutoConsolidationScheduler:
    """One background worker per server, stepping under ``runtime_lock``.

    The pass takes the lock one short step at a time (see ``guard`` on
    ``memory_consolidate``), so recall, lookup and other clients' writes are
    served between its steps instead of after it. A scope a previous process
    left pending is resumed here.
    """

    @contextmanager
    def step() -> Iterator[None]:
        with runtime_lock:
            yield
        # Python locks are not fair: released and re-taken at once, the lock
        # goes back to this thread before a request thread woken by the
        # release gets to run, and recall starves behind a pass made of many
        # short steps. Pausing after each step hands the lock over.
        _time.sleep(_AUTO_CONSOLIDATION_STEP_HANDOFF_SECONDS)

    def run_pass(scope: str) -> dict[str, Any]:
        return _compact_consolidation_summary(
            consolidation_service.memory_consolidate(
                scope=scope,
                force=False,
                guard=step,
            )
        )

    def load_pending() -> str | None:
        with runtime_lock:
            return store.get_kv(AUTO_CONSOLIDATION_PENDING_KV_KEY)

    def save_pending(raw: str) -> None:
        store.set_kv(AUTO_CONSOLIDATION_PENDING_KV_KEY, raw)

    scheduler = AutoConsolidationScheduler(
        run_pass,
        load_pending=load_pending,
        save_pending=save_pending,
        persist_guard=runtime_lock,
    )
    scheduler.resume()
    return scheduler


def _resolve_config(
    *,
    db_path: str | Path | None,
    config: MemoryConfig | None,
    config_path: str | Path | None,
    default_scope: str | None,
) -> MemoryConfig:
    if config is not None and config_path is not None:
        raise ValueError("provide config or config_path, not both")
    if config_path is not None:
        resolved = load_config(config_path)
    else:
        resolved = config or MemoryConfig()
    if db_path is not None:
        resolved = replace(resolved, db_path=Path(db_path))
    if default_scope is not None:
        resolved = replace(resolved, default_scope=default_scope)
    return resolved


def _load_fastmcp() -> Any:
    try:
        from fastmcp import FastMCP
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "FastMCP is required to run the MCP server. Install project dependencies first."
        ) from exc
    return FastMCP


def _configure_tool_list_visibility(
    mcp: Any,
    *,
    expose_attest: bool,
    expose_operator_tools: bool,
) -> None:
    """Filter tool discovery without changing registration or call lookup."""

    add_middleware = getattr(mcp, "add_middleware", None)
    if add_middleware is None:
        # Lightweight test factories expose their registration dictionaries
        # directly and deliberately do not implement the FastMCP runtime.
        return

    from fastmcp.server.middleware import Middleware

    advertised_names = set(_AGENT_TOOL_NAMES)
    if expose_operator_tools:
        advertised_names.update(_OPERATOR_TOOL_NAMES)
    if expose_attest:
        advertised_names.add("memory_attest")

    class _ToolListVisibilityMiddleware(Middleware):
        async def on_list_tools(
            self,
            context: Any,
            call_next: Any,
        ) -> Sequence[Any]:
            tools = await call_next(context)
            return [tool for tool in tools if tool.name in advertised_names]

    add_middleware(_ToolListVisibilityMiddleware())


def _consolidation_result_to_dict(result: ConsolidationResult) -> dict[str, Any]:
    return {
        "concepts_created": [node_to_dict(node) for node in result.concepts_created],
        "concepts_updated": [node_to_dict(node) for node in result.concepts_updated],
        "concepts_promoted": [node_to_dict(node) for node in result.concepts_promoted],
        "schemas_created": [node_to_dict(node) for node in result.schemas_created],
        "schemas_updated": [node_to_dict(node) for node in result.schemas_updated],
        "decayed": [node_to_dict(node) for node in result.decayed],
        "clusters_considered": result.clusters_considered,
        "traces_considered": result.traces_considered,
    }


def _compact_consolidation_summary(result: ConsolidationResult) -> dict[str, Any]:
    """Id-only consolidation report for the memory_remember auto pass.

    memory_consolidate keeps returning full node dicts via
    _consolidation_result_to_dict; ids here resolve through memory_lookup.
    """

    return {
        "concepts_created": [node.id for node in result.concepts_created],
        "concepts_updated": [node.id for node in result.concepts_updated],
        "concepts_promoted": [node.id for node in result.concepts_promoted],
        "schemas_created": [node.id for node in result.schemas_created],
        "schemas_updated": [node.id for node in result.schemas_updated],
        "decayed": [node.id for node in result.decayed],
        "clusters_considered": result.clusters_considered,
        "traces_considered": result.traces_considered,
    }


@dataclass(frozen=True, slots=True)
class _PreparedTraceIndex:
    chunks: tuple[tuple[TextChunk, tuple[float, ...]], ...] = ()
    error: str | None = None


def _prepare_trace_indices(
    store: MemoryStore, contents: Sequence[str],
) -> dict[str, _PreparedTraceIndex]:
    """Prepare exact source bytes without reading knowledge or session state."""

    prepared: dict[str, _PreparedTraceIndex] = {}
    for content in contents:
        try:
            chunks = store._chunk_embeddings_for(content, None, require_complete=True)
            prepared[content] = _PreparedTraceIndex(tuple(
                (chunk, tuple(vector)) for chunk, vector in chunks
            ))
        except Exception as exc:
            prepared[content] = _PreparedTraceIndex(error=type(exc).__name__)
    return prepared


def _index_written_traces(
    store: MemoryStore, nodes: Sequence[Node],
    prepared: Mapping[str, _PreparedTraceIndex],
) -> dict[str, Any] | None:
    """Apply only an exact prepared source inside the required transaction.

    The outer transaction keeps content identity stable and rolls chunks back
    with required knowledge/feedback failures. Each replacement has its own
    savepoint, so a derived-index fault cannot leave a partial chunk set or
    discard knowledge. Missing chunks remain eligible for bounded recall repair.
    """

    deferred = []
    for node in nodes:
        try:
            index = prepared.get(node.content)
            if index is None:
                raise ValueError("prepared trace source does not match stored content")
            error = index.error
            if error is None:
                store.replace_node_chunks(node.id, index.chunks)
        except Exception as exc:
            error = type(exc).__name__
        if error is not None:
            logging.getLogger(__name__).warning("new trace indexing deferred (%s)", error)
            deferred.append(node.id)
    return {"status": "deferred", "node_ids": deferred} if deferred else None


def _remember_node_confirmation(node: Any) -> dict[str, Any]:
    """Compact write receipt for remember and teach: no content echo.

    The writer already holds the content it just stored; provenance bodies
    (prior recall query texts) stay retrievable via memory_lookup(node_id=...).
    """

    provenance = node.provenance or {}
    return {
        "id": node.id,
        "level": node.level,
        "scope": node.scope,
        "created_at": node.created_at,
        "prior_recall_count": len(provenance.get("prior_recalls") or []),
        "source_trace_count": len(node.source_traces or []),
    }


def _teach_result_to_dict(result: TeachResult) -> dict[str, Any]:
    return {
        "corrective_trace": _remember_node_confirmation(result.corrective_trace),
        "supersedes": {"id": result.supersedes.id, "type": result.supersedes.type},
        "original": {"id": result.original.id},
        "implicit_feedback": _implicit_feedback_to_dict(result.implicit_feedback),
    }


def _implicit_feedback_to_dict(feedback: Any) -> dict[str, Any]:
    if feedback is None:
        return {"recall_event_ids": [], "feedback_applied": False}
    return {
        "recall_event_ids": [event.id for event in feedback.events],
        "feedback_applied": feedback.feedback_applied,
    }


def _attach(target: Any, name: str, value: Any) -> None:
    try:
        setattr(target, name, value)
    except Exception:
        return


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run the Living Memory MCP server over one SQLite file.")
    parser.add_argument(
        "sqlite_file",
        nargs="?",
        help="SQLite database file. Defaults to the configured path.",
    )
    parser.add_argument(
        "--db",
        dest="db",
        help="SQLite database file. Overrides the optional positional path.",
    )
    parser.add_argument(
        "--config",
        help="Optional TOML config path.",
    )
    parser.add_argument(
        "--default-scope",
        dest="default_scope",
        help="Fallback scope for new traces when context.scope is omitted. "
             "Also reads LM_DEFAULT_SCOPE or LM_SCOPE env vars (CLI wins).",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "http", "sse"),
        default="stdio",
        help="MCP transport to run.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP or SSE transport.")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP or SSE transport.")
    parser.add_argument(
        "--tls-cert",
        dest="tls_cert",
        help="PEM TLS certificate path. When set together with --tls-key, the "
             "HTTP/SSE transport serves /mcp and admin routes over HTTPS. "
             "Also reads LM_TLS_CERT (CLI wins). Omit both for plain HTTP.",
    )
    parser.add_argument(
        "--tls-key",
        dest="tls_key",
        help="PEM private key path for --tls-cert. Also reads LM_TLS_KEY "
             "(CLI wins). Required together with --tls-cert to enable TLS.",
    )
    parser.add_argument(
        "--embedding",
        dest="embedding",
        help="Embedding backend override (e.g. 'hash', 'online', 'auto'). "
             "Sets LIVING_MEMORY_EMBEDDING_BACKEND for the running server.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
