"""FastMCP server surface for Living Memory."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import deque
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from threading import RLock, Timer
from typing import Any
from uuid import uuid4
import json
import os
import secrets
import sys
import time as _time

from living_memory.config import MemoryConfig, load_config
from living_memory.consolidation import (
    DEFAULT_MIN_CLUSTER_SIZE,
    ConsolidationResult,
    ConsolidationService,
    TeachResult,
)
from living_memory.decay import apply_decay
from living_memory.delivery import (
    DELIVERY_NEAR_DUPLICATE,
    DELIVERY_SESSION_DUPLICATE,
    DELIVERY_TWIN_DUPLICATE,
    context_value_max_chars_from_env,
    full_node_diet_enabled_from_env,
    near_dup_cosine_from_env,
    near_dup_length_ratio_from_env,
    provenance_value_max_chars_from_env,
    session_dedup_enabled_from_env,
    shape_recall_results,
    snippet_ladder_from_env,
    snippet_max_chars_from_env,
    sparse_entries_enabled_from_env,
    stats_compaction_enabled_from_env,
)
from living_memory.edge_derivation import derive_edges_for_new_trace
from living_memory.instructions_map import map_section as _compose_map_section
from living_memory.near_dup import (
    DuplicateCandidate,
    build_duplicate_map,
    identifier_veto_enabled,
    mean_pooled_vectors,
)
from living_memory.prompts import retrieval_context_prompt
from living_memory.recall_map import RecallMapBuilder
from living_memory.resources import (
    connection_to_dict,
    global_concepts,
    memory_health as health_view,
    memory_stats,
    memory_status as status_view,
    node_to_dict,
    project_concepts,
    recent_interactions,
)
from living_memory.retrieval import MemoryRecallService
from living_memory.scope import normalize_scope, resolve_scope
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.feedback import apply_pending_recall_feedback
from living_memory.storage import (
    FingerprintGatePolicy,
    MemoryStore,
    recall_fingerprint,
    should_gate_fingerprint,
)
from living_memory.temporal import parse_timestamp

_BOOT_ID = uuid4().hex
_STARTED_AT = datetime.now(timezone.utc)
_RESTART_PENDING = False

#: Delivery classes a gated response may drop from its trailing run.
#:
#: Every class here is one whose text the agent can already reach without the
#: entry: a session duplicate and a fingerprint repeat were delivered under
#: this same session or this same request moments ago, a twin's bytes ship
#: under a bearer in this very response, and a near-duplicate's meaning does
#: too. ``full`` and ``snippet`` are absent by construction — they are the
#: only classes that carry content nothing else in reach holds.
_DROPPABLE_TRAILING_STUBS = (
    DELIVERY_SESSION_DUPLICATE,
    DELIVERY_TWIN_DUPLICATE,
    DELIVERY_NEAR_DUPLICATE,
)

# Server-wide KV key holding the persisted (rotated) bearer token. A value here
# takes precedence over the LM_AUTH_TOKEN env seed at startup, so a rotation via
# POST /admin/token survives a restart. The token lives only in the git-ignored
# SQLite state file — never logged, echoed in a response, or written to a tracked
# file, so it stays out of logs, diffs, and traces.
_AUTH_TOKEN_KV_KEY = "auth_token"

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
) -> Any:
    """Create a FastMCP server bound to one SQLite store."""

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
        # Composed once here, and again after each recall/remember — see
        # _InstructionsRefresh. Composing at boot rather than only on refresh
        # is what lets a restarted process serve the map its predecessor
        # persisted instead of an empty one.
        "instructions": _instructions_with_map(store, store.config.default_scope),
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
    _register_tools(mcp, store, runtime_lock, expose_attest=expose_attest)
    _register_resources(mcp, store, runtime_lock)
    _register_prompts(mcp, store, runtime_lock)
    _register_admin_routes(
        mcp,
        store=store,
        runtime_lock=runtime_lock,
        auth_state=auth_state,
        default_scope=store.config.default_scope,
    )
    return mcp


def _register_admin_routes(
    mcp: Any,
    *,
    store: MemoryStore,
    runtime_lock: Any,
    auth_state: "_AuthTokenState",
    default_scope: str,
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
                },
                status_code=503,
            )
        return JSONResponse(
            {
                "ok": True,
                "service": "living-memory",
                "boot_id": _BOOT_ID,
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
    # ``map_section`` is spliced at the very TAIL, after the default-scope
    # line, and nowhere else. Clients clip this text at 2048 chars from the
    # FRONT, so a clip lands on whatever sits last: the tail is the only
    # place for the lowest-priority, most volatile content, and the three
    # laws and the bootstrap directive keep the positions that survive any
    # clip. The section is also the only part of this text composed from
    # stored data rather than written here, which is the second reason it
    # goes last — a text that reorders itself between sessions is a text
    # clients cache and agents re-read.
    #
    # An empty (or blank) section returns the static text byte-identical:
    # a store with no map history, a store that cannot answer, and the
    # rolled-back valve all degrade to exactly the text that shipped before
    # any map existed, rather than to a heading with nothing under it.
    #
    # Delivery contract (measured live 2026-08-09, claude-code 2.1.217):
    # clients clip MCP server instructions at 2048 chars (and GigaCode drops
    # them entirely), so this text carries only what must be visible before
    # any tool is considered — identity framing, the three laws, and the
    # bootstrap directive. The detailed binding hooks live in the tool
    # descriptions below (_RECALL_DESCRIPTION etc.), which survive as long
    # as each stays within the per-tool client budgets (1024 chars for
    # OpenAI-compatible clients such as codex). Budgets are pinned by
    # tests/test_instructions_imperative.py.
    #
    # The clip is on the WHOLE string, and the string ends with the
    # default-scope line, so its length grows one-for-one with the scope
    # name: measured 2026-08-20, the pre-compression text was 2042 chars
    # for "global" but already over budget for an ordinary project scope.
    # The connective prose below was therefore reworded tighter — every
    # protocol point still carried, nothing displaced — to leave room for
    # both long scope names and a later tail section. Audited mapping of
    # each rewording to the point (and pinned phrase) it preserves:
    #
    # - "You supply ephemeral reasoning; LM supplies durable memory."
    #   -> folded into the opening clause after the colon; the division of
    #   labour survives, as does the framing pin ("ONE cognitive system",
    #   which must stay inside the first 600 chars).
    # - "`memory_recall` is perception, `memory_remember` is learning, and
    #   `memory_teach` is self-correction." + the three without-X clauses
    #   -> merged so each role and its consequence is stated once instead
    #   of twice. "inventing what memory already holds" survives
    #   byte-identical; the four tool-name pins are carried by the
    #   bootstrap paragraph (and law 3), never by this sentence.
    # - "This is not a tool for when convenient — it is how you think:
    #   recall is perception, not an extra step." -> dropped as a
    #   restatement: "ONE cognitive system", "Recall is perception" and
    #   the MUST register of the three laws already carry it.
    # - Law 1's trigger enumeration (change, irreversible step, knowledge
    #   search, design choice) -> the broader "Every action, decision";
    #   the exact enumeration lives in _RECALL_DESCRIPTION, which is the
    #   binding channel for triggers. "You MUST recall BEFORE you act" and
    #   "When uncertain or stuck, recall" survive byte-identical.
    # - "a miss costs one tool call; skipping can repeat an entire
    #   debugging cycle." -> "a miss costs one call; skipping repeats
    #   whole cycles of work." — same economics, no longer debugging-only.
    # - Law 2's "Sessions end without warning — a deferred remember is
    #   usually lost." -> same clause as a subordinate; the loss
    #   consequence, not just the timing, still lands.
    # - Law 3's tail "a correction stored as a remember trace keeps losing
    #   to the stale node in every future recall." -> "a remember trace
    #   keeps losing to the stale node."; "supersedes edge" survives.
    # - Bootstrap "(exact BEFORE triggers)" -> "(BEFORE triggers)". Every
    #   pin in this paragraph ("load the Living Memory tools NOW", "recall
    #   the task at hand", "as protocol", "defers", the four tool names)
    #   is untouched.
    static = (
        "You and Living Memory form ONE cognitive system: you supply "
        "ephemeral reasoning, LM durable memory. Recall is perception — "
        "without it you act blind, inventing what memory already holds; "
        "remember is learning, teach is self-correction — without them "
        "the system loses what you learn and repeats corrected "
        "mistakes.\n"
        "\n"
        "## Three laws — always in force\n"
        "\n"
        "1. You MUST recall BEFORE you act. Every action, decision, new "
        "task, and new turn of thought has history here. When uncertain "
        "or stuck, recall: a miss costs one call; skipping repeats whole "
        "cycles of work.\n"
        "2. You MUST remember at the moment of insight, before your next "
        "action — sessions end without warning, and a deferred remember "
        "is lost. One concrete fact per trace.\n"
        "3. You MUST teach the moment a belief changes. When a recalled "
        "fact proves wrong, call `memory_teach` at once — only teach "
        "creates the supersedes edge; a remember trace keeps losing to "
        "the stale node.\n"
        "\n"
        "## The full protocol lives on the tools\n"
        "\n"
        "The binding trigger protocol is in the tool descriptions "
        "themselves: `memory_recall` (BEFORE triggers), `memory_remember` "
        "(write policy and closure notes), `memory_teach` (correction "
        "rules), `memory_consolidate` (promotion to schemas). If this "
        "session defers or hides tool schemas behind a search step, you "
        "MUST load the Living Memory tools NOW, at session start, recall "
        "the task at hand, and follow their descriptions as protocol, "
        "not as reference.\n"
        "\n"
        "## Scope and context\n"
        "\n"
        "Write with a specific scope: `context.scope='project:<name>'` for "
        "project facts, `global` for universal rules and procedural "
        "schemas. Read broad: omit `scope` when investigating, so "
        "cross-project lessons transfer. Set `task`, `agent`, and "
        "`session_id` in context to link work across sessions.\n"
        f"Your default scope is {default_scope}.\n"
    )
    section = map_section.strip()
    if not section:
        return static
    return f"{static}\n{section}\n"


#: History rows the section is composed from, far below
#: ``instructions_map.DEFAULT_HISTORY_LIMIT``. Two reasons, and the cheap one
#: is not the main one.
#:
#: Recency is the point: this channel personalizes a session from what memory
#: has *lately* been mapping, and a window reaching twenty recalls back drags
#: in structure the work has moved on from. Three maps carry up to three times
#: ``instructions_map.MAX_LABELS`` candidate labels for the six slots the
#: section shows, so the cap binds long before the window does.
#:
#: It is also what keeps the refresh off the latency budget. Composition
#: screens every label of every cluster of every row against the register ban
#: battery, so its cost is linear in rows x clusters and it — not the indexed
#: read — is what a wide window costs: measured 2026-08-20 over a 40-map
#: history, twenty rows cost 0.14ms of read against 0.54ms of composition,
#: and a whole refresh falls from 0.68ms there to 0.13ms at three.
_INSTRUCTIONS_MAP_HISTORY_ROWS = 3


def _instructions_with_map(store: MemoryStore, default_scope: str) -> str:
    """The instructions text this store's persisted history composes right now.

    Called once at construction and again after each recall/remember, never
    on a path an agent waits on for anything else. The read is one bounded,
    indexed query and the composition is pure — no LLM, no network — because
    this text is also what ``initialize`` blocks on.

    ``LM_RECALL_MAP=0`` removes the section as well as the response key: the
    valve is the rollback for the whole map, and a rollback that left one
    channel populated would not be one.
    """

    section = ""
    if _recall_map_enabled():
        try:
            section = _compose_map_section(
                store, limit=_INSTRUCTIONS_MAP_HISTORY_ROWS
            )
        except Exception:
            # The instructions are the one text every client shows before any
            # tool is considered. They render, whatever the store says.
            section = ""
    return _server_instructions(default_scope, map_section=section)


class _InstructionsRefresh:
    """Re-snapshot the served instructions between transport sessions.

    ``FastMCP`` takes ``instructions`` as a construction kwarg and never
    re-reads the composed text, so nothing here would ever change after boot.
    What *is* re-read is the low-level attribute: ``Server.
    create_initialization_options()`` reads ``instructions=self.instructions``
    at call time, and ``StreamableHTTPSessionManager`` calls it inside each
    ``app.run(...)`` — once per transport session. Assigning
    ``_mcp_server.instructions`` therefore reaches the NEXT session's
    ``initialize``; the current session, already initialized, is untouched.
    Over stdio there is one session per process, so the map a session sees is
    the state at its own boot.

    Two properties this must have, because it hangs off the recall path:

    * It never raises into the tool. A map is a decoration on instructions;
      losing it must never lose a recall or a write.
    * It is defensive about the object. ``create_mcp_server`` accepts an
      ``mcp_factory``, and a test double is under no obligation to own a
      ``_mcp_server`` — absent, this is a silent no-op.

    The assignment is skipped when the composed text is unchanged, which is
    the common case: history only moves when a recall delivers a new map.
    """

    def __init__(self, mcp: Any, store: MemoryStore) -> None:
        self._mcp = mcp
        self._store = store
        self._default_scope = store.config.default_scope
        # Whatever was composed at construction, so the first call assigns
        # only if the history has actually moved since boot.
        self._text = getattr(mcp, "instructions", None)

    def __call__(self) -> None:
        try:
            server = getattr(self._mcp, "_mcp_server", None)
            if server is None:
                return
            text = _instructions_with_map(self._store, self._default_scope)
            if text == self._text:
                return
            self._text = text
            server.instructions = text
        except Exception:
            return


# MCP-visible tool descriptions. These are the binding protocol channel:
# unlike server instructions they reach every client that can call tools
# at all (including GigaCode, which drops instructions). Each MUST stay
# within 1024 chars — OpenAI-compatible clients (codex) enforce that per
# function description; claude-code clips around 2048. Front-load the
# imperative triggers: anything past the budget is the first to be lost.

_RECALL_DESCRIPTION = (
    "Retrieve memories by text, vector, and graph signals. "
    "You MUST recall BEFORE acting: "
    "changing any artifact (conventions, rejected approaches are "
    "stored); creating or mutating state (recall the domain concept "
    "before inventing one); irreversible or outward-facing steps "
    "(recall action plus target); pulling knowledge from the world — "
    "searching, measuring (memory first: the world only after recall "
    "returns nothing — anti-pattern: world-before-memory); choosing a "
    "design or approach (recall cross-project); entering anything new "
    "— session, task, message. "
    "And you MUST recall MID-WORK, at every new turn of thought: ask "
    "what memory holds nearby before the turn hardens, so nothing "
    "related is missed; stuck or surprised means overdue, 'if only I "
    "knew' means recall NOW. "
    "Default: when uncertain, recall. "
    "Read broad: omit scope to transfer across scopes. Depth 'causal' "
    "when debugging. A level:schema result is a binding procedure — "
    "follow it literally. Non-full results carry a content_ref — "
    "refetch via memory_lookup."
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
    # One builder per server, like the service's own caches: the map's
    # stability guarantee is "same task, same structure across sessions",
    # which only holds if the thing remembering the structure outlives a call.
    recall_map_builder = RecallMapBuilder(store)
    consolidation_service = ConsolidationService(store)
    # Runs under ``runtime_lock`` with the rest of the tool body: the store
    # holds one sqlite connection shared across threads, and the refresh reads
    # it. One bounded indexed query, after the response is already built.
    refresh_instructions = _InstructionsRefresh(mcp, store)

    @mcp.tool(description=_REMEMBER_DESCRIPTION)
    @_track_latency("memory_remember")
    def memory_remember(
        content: str,
        context: dict[str, Any] | None = None,
        feedback: dict[str, Any] | None = None,
        alternatives_considered: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Store a new append-only trace.

        Returns a compact confirmation — node id/level/scope plus feedback and
        consolidation counters. Fetch full nodes via memory_lookup(node_id=...).
        """

        context = _with_transport_identity(context)
        with runtime_lock:
            try:
                auto_decay = _decay_sweep_if_due(store)
            except Exception:
                auto_decay = None
            if alternatives_considered is None:
                node = store.append_trace(content, context, feedback=feedback)
                rejected_alternatives: list[Any] | None = None
            else:
                node, rejected_nodes = store.append_trace_with_rejected_alternatives(
                    content,
                    context,
                    feedback=feedback,
                    alternatives_considered=alternatives_considered,
                )
                rejected_alternatives = [rejected.id for rejected in rejected_nodes]
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
            auto_consolidation = _auto_consolidate_if_due(
                store,
                consolidation_service,
                node.scope,
            )
            response = {
                "node": _remember_node_confirmation(node),
                "implicit_feedback": _implicit_feedback_to_dict(implicit_feedback),
                "auto_consolidation": auto_consolidation,
                "auto_decay": auto_decay,
            }
            if rejected_alternatives is not None:
                response["rejected_alternatives"] = rejected_alternatives
            refresh_instructions()
            return response

    @mcp.tool(description=_TEACH_DESCRIPTION)
    @_track_latency("memory_teach")
    def memory_teach(
        trace_id: str,
        correction: str | dict[str, Any],
        confidence: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a corrective trace and connect it to the original."""

        context = _with_transport_identity(context)
        with runtime_lock:
            taught = consolidation_service.memory_teach(
                trace_id,
                correction,
                confidence=confidence,
                context=context,
            )
            return _teach_result_to_dict(taught)

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

    @mcp.tool(description=_RECALL_DESCRIPTION)
    @_track_latency("memory_recall")
    def memory_recall(
        query: str,
        scope: str | None = None,
        depth: int | str | None = 1,
        max_results: int = 5,
        ambient_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant memory nodes using scope, text, vector, and graph signals.

        Each result carries a ``delivery`` class: ``full``, ``snippet`` (long
        content truncated inline), ``session_duplicate`` (already delivered on
        this transport session), ``twin_duplicate`` (byte-identical to a
        higher-ranked result), or ``near_duplicate`` (a paraphrase of a
        higher-ranked result of this same answer). Every class is on by
        default: twin and near-dup collapse unconditionally, the session class
        whenever a transport session id is stamped. The near-dup threshold is
        ``LM_RECALL_NEAR_DUP_COSINE`` (0 restores byte-only dedup).
        The top-ranked content-bearer ships complete content;
        lower-ranked bearers get descending snippet budgets.
        Non-full results add a ``content_ref`` whose ``node_id`` fed to
        ``memory_lookup`` returns the complete stored node. Every delivery is
        dieted on the wire — oversized context/provenance values compacted to
        counts/truncations, bookkeeping stats and zero/null fields dropped —
        and ``memory_lookup`` restores the stored node byte-complete
        (``LM_DELIVERY_*`` env valves roll each cut back).

        When the ranked candidate pool holds more than the delivered results,
        an additive top-level ``recall_map`` describes the *residual*: a
        handful of labelled clusters with counts, a medoid example and a
        phrasing to ask with — what else memory holds for this task, as a plan
        for the next recall. It never reorders or replaces ``results``, and it
        is absent whenever there is no residual or nothing worth saying about
        it (``LM_RECALL_MAP=0`` removes it outright).
        """

        ambient_context = _with_transport_identity(ambient_context)
        raw_transport_id = (ambient_context or {}).get(_TRANSPORT_SESSION_KEY)
        # Mirror storage's stamping (str() of any non-None value) so the dedup
        # probe compares equal to the recall_events.transport_session_id column.
        transport_session_id = str(raw_transport_id) if raw_transport_id is not None else None
        session_dedup = bool(transport_session_id) and session_dedup_enabled_from_env()
        with runtime_lock:
            auto_decay = _maybe_decay_sweep(store)
            # Snapshot delivered ids BEFORE the service records this call's
            # recall_event, so the current response cannot stub itself.
            already_delivered = (
                store.delivered_node_ids(transport_session_id)
                if session_dedup
                else set()
            )
            # Experimental repeat gating is a strict opt-in via
            # LM_RECALL_REPEAT_GATING=1. resolve_scope runs the same resolver
            # retrieval will, so the fingerprint equals the one
            # record_recall_event stamps on this event; stats are read before
            # the service records this delivery — the decision sees only the
            # accounting accumulated by prior requests.
            gate_policy = FingerprintGatePolicy.from_env()
            gated = False
            if gate_policy.enabled:
                plan = resolve_scope(
                    query=query,
                    scope=scope,
                    ambient_context=ambient_context,
                    store=store,
                )
                fingerprint = recall_fingerprint(query, plan.requested_scope)
                gated = should_gate_fingerprint(
                    store.get_recall_fingerprint_stats(fingerprint), gate_policy
                )
                if gated:
                    # Nodes this exact request already shipped become
                    # session_duplicate stubs; never-delivered nodes still
                    # ship content-bearing.
                    already_delivered = already_delivered | (
                        store.fingerprint_delivered_node_ids(fingerprint)
                    )
            results = recall_service.memory_recall(
                query,
                scope=scope,
                ambient_context=ambient_context,
                depth=depth,
                max_results=max_results,
            )
            if gated and recall_service.last_recall_event_id is not None:
                store.mark_recall_event_gated(recall_service.last_recall_event_id)
            shaped = shape_recall_results(
                results,
                already_delivered_ids=already_delivered,
                snippet_max_chars=snippet_max_chars_from_env(),
                context_value_max_chars=context_value_max_chars_from_env(),
                session_dedup=session_dedup or gated,
                snippet_ladder=snippet_ladder_from_env(),
                full_node_diet=full_node_diet_enabled_from_env(),
                provenance_value_max_chars=provenance_value_max_chars_from_env(),
                stats_compaction=stats_compaction_enabled_from_env(),
                sparse_entries=sparse_entries_enabled_from_env(),
                # Built here, from this answer's own node vectors, because
                # shaping is pure: the database read and the env threshold
                # live on this side of the call.
                duplicate_of=_near_duplicate_map(store, results),
            )
            if gated and gate_policy.drop_trailing_stubs:
                # The trailing all-stub run of a gated delivery carries no
                # content the agent is not already holding, yet stub entries
                # still cost ~1.1k chars each, so the gate drops the run (the
                # reduced-result-count form of compaction). The independent
                # LM_RECALL_REPEAT_DROP_TRAILING_STUBS=1 opt-in enables this;
                # otherwise the stub list remains. Content-bearers and stubs
                # ranked above them survive, and the recorded recall_event
                # keeps every result id.
                while shaped and shaped[-1]["delivery"] in _DROPPABLE_TRAILING_STUBS:
                    shaped.pop()
            response: dict[str, Any] = {
                "query": query,
                "scope": scope,
                "recall_event_id": recall_service.last_recall_event_id,
                "count": len(shaped),
                "results": shaped,
                "auto_decay": auto_decay,
            }
            # The map of what the cut left behind, attached verbatim as one
            # more top-level key — the same additive shape ``auto_decay``
            # established. Everything about *what* it says lives in
            # recall_map.py: the server decides only whether there is a
            # residual to describe, and records what it delivered.
            residual = recall_service.last_residual
            if residual and _recall_map_enabled():
                built = recall_map_builder.build(
                    residual,
                    scope=scope,
                    task=_ambient_text(ambient_context, "task"),
                    task_pattern=_ambient_text(ambient_context, "task_pattern"),
                )
                if built is not None:
                    response["recall_map"] = built.to_dict()
                    if recall_service.last_recall_event_id is not None:
                        store.attach_recall_map(
                            recall_service.last_recall_event_id,
                            response["recall_map"],
                        )
            # The map this call just persisted is the signal the *next*
            # session's instructions are composed from. Last, after the
            # response is complete, so nothing an agent waits on depends on
            # it — and swallowing, so nothing it can hit costs a recall.
            refresh_instructions()
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


    @mcp.tool
    @_track_latency("memory_lookup")
    def memory_lookup(
        scope: str | None = None,
        task_pattern: str | None = None,
        procedure_id: str | None = None,
        lesson_kind: str | None = None,
        level: str | None = "trace",
        node_id: str | None = None,
        node_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return memory nodes by exact context-field match, or full nodes by id.

        ``node_id``/``node_ids`` fetch complete nodes directly — the re-fetch
        path for snippet/duplicate recall stubs. Missing ids are skipped and
        listed under ``missing``; decayed nodes are returned with their
        ``decayed`` flag set; ``level`` is ignored for id fetches. A ``scope``
        passed alongside ids verifies instead of filtering: mismatching nodes
        stay in ``results`` and are reported in ``scope_mismatches``.
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
                found = store.get_nodes(requested_ids)
                results = [
                    node_to_dict(found[requested])
                    for requested in requested_ids
                    if requested in found
                ]
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
                return response

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
                "results": [node_to_dict(node) for node in nodes],
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


_ADAPTIVE_MIN_TRACES = 5


def _adaptive_trigger_step(trace_count: int) -> int | None:
    """Trigger step for adaptive auto-consolidate policy.

    Fires earlier on young scopes (every 5 up to 50, every 25 up to 500)
    and falls back to the fixed cadence on mature scopes. Returns None when
    the scope holds too few traces to be worth consolidating.
    """

    if trace_count < _ADAPTIVE_MIN_TRACES:
        return None
    if trace_count < 50:
        return 5
    if trace_count < 500:
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


def _near_duplicate_map(
    store: MemoryStore, results: Sequence[Any]
) -> dict[str, str] | None:
    """Build this answer's near-duplicate map, or None when there is none.

    The impure half of the near-dup collapse, deliberately kept here rather
    than in ``delivery``: it reads the env threshold and it reads the database.
    ``shape_recall_results`` then receives a finished ``{duplicate: bearer}``
    map and stays the pure function it is documented to be.

    Only the handful of node ids this answer actually returns are pooled, so
    the read is ``max_results`` chunk lookups (5 by default) and not a matrix
    over the scope. A node with no chunk rows — never drained, or written
    since the last drain — is simply absent from ``vectors`` and therefore
    never collapsed.

    ``LM_RECALL_NEAR_DUP_COSINE=0`` returns before the first query: the
    rollback valve costs one env read, not one wasted database pass.

    Each candidate carries its text, because the identifier veto reads it: a
    result naming an id, path or node name its bearer does not name is a
    different fact and must not become a stub. ``LM_NEAR_DUP_IDENTIFIER_VETO``
    is that valve, read here and in the drain through the same function.
    """

    if not results:
        return None
    cosine_threshold = near_dup_cosine_from_env()
    if cosine_threshold <= 0.0:
        return None
    vectors = mean_pooled_vectors(store, [result.node.id for result in results])
    if not vectors:
        return None
    duplicate_of = build_duplicate_map(
        [
            DuplicateCandidate(
                node_id=result.node.id,
                content_length=len(result.node.content or ""),
                content=result.node.content or "",
            )
            for result in results
        ],
        vectors,
        cosine_threshold=cosine_threshold,
        min_length_ratio=near_dup_length_ratio_from_env(),
        identifier_veto=identifier_veto_enabled(),
    )
    return duplicate_of or None


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


def _auto_consolidate_if_due(
    store: MemoryStore,
    consolidation_service: ConsolidationService,
    scope: str,
) -> dict[str, Any] | None:
    row = store.connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM nodes
        WHERE level = 'trace' AND scope = ? AND decayed = 0
        """,
        (scope,),
    ).fetchone()
    trace_count = int(row["count"])

    if _auto_consolidate_policy() == "adaptive":
        step = _adaptive_trigger_step(trace_count)
        if step is None or trace_count % step != 0:
            return None
        # The merge floor itself is no longer decided here: consolidation.py
        # owns the ladder and the LM_CONSOLIDATE_MERGE_FLOOR valve, and sizes
        # it from this same scope's active traces. This branch only picks the
        # trigger cadence.
        return _compact_consolidation_summary(
            consolidation_service.memory_consolidate(scope=scope, force=False)
        )

    if trace_count < DEFAULT_MIN_CLUSTER_SIZE or trace_count % DEFAULT_MIN_CLUSTER_SIZE != 0:
        return None
    return _compact_consolidation_summary(
        consolidation_service.memory_consolidate(scope=scope, force=False)
    )


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


def _remember_node_confirmation(node: Any) -> dict[str, Any]:
    """Compact write confirmation: identity and counters, no content echo.

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
        "linked_node_count": len(provenance.get("recalled_nodes") or []),
        "source_trace_count": len(node.source_traces or []),
    }


def _teach_result_to_dict(result: TeachResult) -> dict[str, Any]:
    return {
        "corrective_trace": node_to_dict(result.corrective_trace),
        "supersedes": connection_to_dict(result.supersedes),
        "original": node_to_dict(result.original),
        "implicit_feedback": _implicit_feedback_to_dict(result.implicit_feedback),
    }


def _implicit_feedback_to_dict(feedback: Any) -> dict[str, Any]:
    if feedback is None:
        return {"recall_event_ids": [], "linked_node_ids": [], "feedback_applied": False}
    return {
        "recall_event_ids": [event.id for event in feedback.events],
        "linked_node_ids": list(feedback.linked_node_ids),
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
