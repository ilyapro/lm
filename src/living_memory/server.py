"""FastMCP server surface for Living Memory."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import deque
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
from living_memory.prompts import retrieval_context_prompt
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
from living_memory.retrieval import MemoryRecallService, RecallResult
from living_memory.scope import normalize_scope
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.feedback import apply_pending_recall_feedback
from living_memory.storage import MemoryStore
from living_memory.temporal import parse_timestamp

_BOOT_ID = uuid4().hex
_STARTED_AT = datetime.now(timezone.utc)
_RESTART_PENDING = False

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
        "instructions": _server_instructions(store.config.default_scope),
    }
    token_value = auth_token or ""
    token_value = token_value.strip()
    # Only wire auth on the real FastMCP — custom test factories don't accept it.
    if token_value and mcp_factory is None:
        auth_provider = _build_static_token_auth(token_value)
        if auth_provider is not None:
            factory_kwargs["auth"] = auth_provider
    mcp = mcp_cls(name, **factory_kwargs)
    runtime_lock = RLock()
    _attach(mcp, "memory_store", store)
    _register_tools(mcp, store, runtime_lock)
    _register_resources(mcp, store, runtime_lock)
    _register_prompts(mcp, store, runtime_lock)
    _register_admin_routes(
        mcp,
        store=store,
        runtime_lock=runtime_lock,
        expected_token=token_value,
        default_scope=store.config.default_scope,
    )
    return mcp


def _register_admin_routes(
    mcp: Any,
    *,
    store: MemoryStore,
    runtime_lock: Any,
    expected_token: str,
    default_scope: str,
) -> None:
    """Register /health, /admin/info, /admin/restart, /admin/decay-sweep on FastMCP."""

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
        return bool(expected_token) and secrets.compare_digest(
            _bearer_token(request),
            expected_token,
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


def _build_static_token_auth(token: str) -> Any | None:
    """Return a FastMCP static-token verifier or None if unsupported by factory."""

    try:
        from fastmcp.server.auth import StaticTokenVerifier
    except ImportError:
        return None
    return StaticTokenVerifier(
        tokens={token: {"client_id": "living-memory", "scopes": []}},
    )


def _server_instructions(default_scope: str) -> str:
    return (
        "You and Living Memory form ONE cognitive system. You supply "
        "ephemeral reasoning; LM supplies durable memory. `memory_recall` "
        "is perception, `memory_remember` is learning, and `memory_teach` "
        "is self-correction. Without recall you act blind to prior "
        "experience; without remember you lose what you just learned. "
        "The system grows ONLY through consistent recall/remember/teach "
        "cycles.\n"
        "\n"
        "This is not a tool you reach for when convenient. It is part of "
        "how you think. Treat recall as perception, not an extra step.\n"
        "\n"
        "## Imperative protocol — action-lifecycle hooks\n"
        "\n"
        "### BEFORE Edit / Write / NotebookEdit on any file\n"
        "You MUST recall before touching a file. Pass the file path, "
        "module name, or feature name to `memory_recall`. Prior edits, "
        "conventions, and rejected approaches are stored.\n"
        "\n"
        "### BEFORE Bash that writes state (mkdir, echo > file, sed -i, tee)\n"
        "You MUST recall before creating directories or files that hold "
        "state. If a service API owns the state, recall the domain noun "
        "(e.g. 'goals state', 'session metadata') first.\n"
        "\n"
        "### BEFORE destructive or external Bash (rm, git push, curl POST, "
        "gh pr merge, DROP, ALTER, migrate)\n"
        "You MUST recall before executing — pass the verb plus target "
        "(e.g. 'git push origin master', 'DROP TABLE users'). Past "
        "incidents on the same operation live in LM.\n"
        "\n"
        "### BEFORE codebase grep / find / rg / ast-grep for project knowledge\n"
        "You MUST recall before grepping. LM answers domain questions "
        "and records prior reasoning the code does not show. Grep only "
        "after recall returns nothing.\n"
        "\n"
        "### BEFORE proposing an architectural decision or approach\n"
        "You MUST recall cross-project for the same problem class. "
        "Patterns from `project:alpha` apply to `project:beta`; find the "
        "precedent before designing from scratch.\n"
        "\n"
        "### BEFORE starting a new task or answering a domain question\n"
        "You MUST recall project and task context at session cold start "
        "and before answering 'how does X work?' questions. Without this, "
        "domain answers are invented instead of retrieved.\n"
        "\n"
        "### AFTER discovering non-obvious behaviour, a root cause, or a "
        "working fix\n"
        "You MUST remember after the discovery, at the moment of insight. "
        "Deferred remembers are usually lost. Store one concrete fact per "
        "trace, with file paths and identifiers.\n"
        "\n"
        "### AFTER a failure resolution or pattern recognition\n"
        "You MUST remember after resolving the failure — store the "
        "failure mode and fix. If a recalled fact was wrong, you MUST "
        "call `memory_teach` with the original trace ID and correction.\n"
        "\n"
        "### AFTER a user correction or preference statement\n"
        "When the user overrides your approach or states a preference, "
        "you MUST remember it immediately. Store the corrected behaviour "
        "and the triggering context.\n"
        "\n"
        "### Default: when uncertain, recall\n"
        "The hooks above are a minimum, not a ceiling. When uncertain, "
        "recall. A recall miss costs one tool call; skipping recall can "
        "repeat an entire debugging cycle.\n"
        "\n"
        "### Default: when you learn something, remember\n"
        "Any durable insight warrants `memory_remember`. If you think "
        "'I wish I had known this earlier', store it now. Do not wait "
        "for end-of-task.\n"
        "\n"
        "### Default: when a belief changes, teach\n"
        "`memory_teach` is not only for correcting recalled facts. Any "
        "time your understanding shifts and a prior trace reflects the "
        "old belief, teach the correction. The system cannot self-correct "
        "silently. NEVER record a correction as a plain `memory_remember` "
        "trace with UPDATE/CORRECTION wording: only `memory_teach` "
        "creates the supersedes edge, and without that edge the stale "
        "node keeps outranking your correction in future recalls.\n"
        "\n"
        "## Procedural skill activation (`level:schema`)\n"
        "\n"
        "When `memory_recall` returns a trace tagged `level:schema`, it "
        "is a procedural skill, not a suggestion. You MUST follow the "
        "procedure literally; schemas exist to prevent repeated failure "
        "modes.\n"
        "\n"
        "## Anti-patterns — DO NOT repeat these mistakes\n"
        "\n"
        "1. **mkdir-vs-API**: writing state with `mkdir` / `echo >` / "
        "`tee` when a service API exists is wrong; recall first and use "
        "the API endpoint.\n"
        "2. **shell-watchdog-vs-loop**: do not write a `*_watchdog.sh` "
        "loop when a structured loop primitive exists; recall the "
        "watch/poll pattern and use the primitive.\n"
        "3. **grep-before-recall**: running `grep` / `rg` / `find` for "
        "domain knowledge before `memory_recall` is a mistake; recall "
        "first.\n"
        "4. **silent-correction**: fixing a wrong assumption without "
        "`memory_teach` leaves stale memory active — a `memory_remember` "
        "trace with UPDATE/CORRECTION wording does not count. Every wrong "
        "recall MUST trigger `memory_teach` with the corrected fact.\n"
        "5. **lookup-table-as-learning**: storing a universal rule as a "
        "project-local note makes the system re-invent it. Store "
        "universal rules in `global` and let `memory_consolidate` promote "
        "schemas.\n"
        "\n"
        "## Cross-project knowledge transfer\n"
        "\n"
        "Patterns generalise across scopes. You MUST broaden recall "
        "(omit `scope` or pass a wildcard) when investigating a class of "
        "problem so cross-project lessons transfer. Keep write-time "
        "scope specific; keep read-time recall broad.\n"
        "\n"
        "## Scope at write time\n"
        "\n"
        "Set `context.scope` to `'project:<name>'` for project-specific "
        "facts. Use `global` for universal debugging patterns, coding "
        "rules, and procedural schemas. Scope is inferred when omitted; "
        "explicit is better.\n"
        f"Your default scope is {default_scope}.\n"
        "\n"
        "## Structured context — required fields\n"
        "\n"
        "`memory_recall.ambient_context` and `memory_remember.context` MUST "
        "pass matching structured metadata:\n"
        "```json\n"
        '{"scope": "project:online", "task": "EZ-13771", "agent": "codex", "session_id": "run-2026-05-22"}\n'
        "```\n"
        "Without matching `task`, `agent`, and `session_id`, recalls "
        "cannot reliably link to later remember/teach feedback.\n"
        "\n"
        "## How to write good traces\n"
        "\n"
        "- Concrete: prefer 'file X exports Y, not Z' over vague notes.\n"
        "- Specific: include file paths, function names, config keys.\n"
        "- One fact per trace. Short beats long.\n"
        "- Include WHY when non-obvious.\n"
        "- Use `depth: 'causal'` on `memory_recall` when debugging.\n"
        "\n"
        "## What NOT to store\n"
        "\n"
        "- Routine actions ('ran the tests') without new insight.\n"
        "- Copies of code — reference file paths instead.\n"
        "- Speculation or unverified plans — store verified facts only.\n"
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


def _register_tools(mcp: Any, store: MemoryStore, runtime_lock: Any) -> None:
    recall_service = MemoryRecallService(store)
    consolidation_service = ConsolidationService(store)

    @mcp.tool
    @_track_latency("memory_remember")
    def memory_remember(
        content: str,
        context: dict[str, Any] | None = None,
        feedback: dict[str, Any] | None = None,
        alternatives_considered: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Store a new append-only trace."""

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
            auto_consolidation = _auto_consolidate_if_due(
                store,
                consolidation_service,
                node.scope,
            )
            response = {
                "node": node_to_dict(node),
                "implicit_feedback": _implicit_feedback_to_dict(implicit_feedback),
                "auto_consolidation": auto_consolidation,
                "auto_decay": auto_decay,
            }
            if rejected_alternatives is not None:
                response["rejected_alternatives"] = rejected_alternatives
            return response

    @mcp.tool
    @_track_latency("memory_teach")
    def memory_teach(
        trace_id: str,
        correction: str | dict[str, Any],
        confidence: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a corrective trace and connect it to the original."""

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

    @mcp.tool
    @_track_latency("memory_recall")
    def memory_recall(
        query: str,
        scope: str | None = None,
        depth: int | str | None = 1,
        max_results: int = 10,
        ambient_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant memory nodes using scope, text, vector, and graph signals."""

        with runtime_lock:
            auto_decay = _maybe_decay_sweep(store)
            results = recall_service.memory_recall(
                query,
                scope=scope,
                ambient_context=ambient_context,
                depth=depth,
                max_results=max_results,
            )
            return {
                "query": query,
                "scope": scope,
                "recall_event_id": recall_service.last_recall_event_id,
                "count": len(results),
                "results": [_recall_result_to_dict(result) for result in results],
                "auto_decay": auto_decay,
            }

    @mcp.tool
    @_track_latency("memory_lookup")
    def memory_lookup(
        scope: str,
        task_pattern: str | None = None,
        procedure_id: str | None = None,
        lesson_kind: str | None = None,
        level: str | None = "trace",
    ) -> dict[str, Any]:
        """Return memory nodes by exact context-field match without ranked recall."""

        with runtime_lock:
            normalized_scope = normalize_scope(scope)
            filters = {
                key: value
                for key, value in {
                    "task_pattern": task_pattern,
                    "procedure_id": procedure_id,
                    "lesson_kind": lesson_kind,
                }.items()
                if value is not None
            }
            if not filters:
                return {
                    "error": "at least one exact context filter is required",
                    "scope": normalized_scope,
                    "filters": {},
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

    @mcp.tool
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


def _adaptive_merge_floor(trace_count: int) -> int:
    """Cluster size required to promote into a concept under adaptive policy."""

    if trace_count < 25:
        return 3
    if trace_count < 100:
        return 5
    if trace_count < 1000:
        return 25
    return DEFAULT_MIN_CLUSTER_SIZE


def _auto_consolidate_policy() -> str:
    return os.environ.get("LM_AUTO_CONSOLIDATE_POLICY", "fixed").strip().lower()


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
        return _consolidation_result_to_dict(
            consolidation_service.memory_consolidate(
                scope=scope,
                force=False,
                min_cluster_size=_adaptive_merge_floor(trace_count),
            )
        )

    if trace_count < DEFAULT_MIN_CLUSTER_SIZE or trace_count % DEFAULT_MIN_CLUSTER_SIZE != 0:
        return None
    return _consolidation_result_to_dict(
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


def _recall_result_to_dict(result: RecallResult) -> dict[str, Any]:
    return {
        "node": node_to_dict(result.node),
        "score": result.score,
        "bm25_score": result.bm25_score,
        "vector_score": result.vector_score,
        "graph_score": result.graph_score,
        "trigger_score": result.trigger_score,
        "scope_rank": result.scope_rank,
        "methods": list(result.methods),
        "path": list(result.path),
        "recall_event_id": result.recall_event_id,
    }


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
