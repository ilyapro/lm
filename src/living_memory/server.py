"""FastMCP server surface for Living Memory."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock, Timer
from typing import Any
from uuid import uuid4
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
from living_memory.feedback import apply_pending_recall_feedback
from living_memory.storage import MemoryStore
from living_memory.temporal import parse_timestamp

_BOOT_ID = uuid4().hex
_STARTED_AT = datetime.now(timezone.utc)
_RESTART_PENDING = False


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
        "ephemeral in-context reasoning; LM supplies durable persistent "
        "memory. `memory_recall` is the system's perception of its past; "
        "`memory_remember` is the system's learning step; `memory_teach` "
        "is its self-correction. Without recall you act blind to prior "
        "experience. Without remember you lose the insight you just "
        "earned. The system grows in intelligence ONLY through consistent "
        "recall/remember/teach cycles — skip them and the learning loop "
        "breaks; future-you loses what you learned.\n"
        "\n"
        "This is not a tool you reach for when convenient. It is part of "
        "how you think. Treat recall as perception, not an extra step.\n"
        "\n"
        "## Imperative protocol — action-lifecycle hooks\n"
        "\n"
        "### BEFORE Edit / Write / NotebookEdit on any file\n"
        "You MUST recall before touching a file. Pass the file path, "
        "module name, or feature name to `memory_recall`. Prior edits, "
        "conventions, and rejected approaches are stored. A pattern you "
        "re-discover by grepping is a pattern you should have recalled.\n"
        "\n"
        "### BEFORE Bash that writes state (mkdir, echo > file, sed -i, tee)\n"
        "You MUST recall before creating directories or files that hold "
        "state. Writing state to the filesystem when a service API "
        "exists is a recurring failure mode (see anti-patterns). Recall "
        "the domain noun (e.g. 'goals state', 'session metadata') first.\n"
        "\n"
        "### BEFORE destructive or external Bash (rm, git push, curl POST, "
        "gh pr merge, DROP, ALTER, migrate)\n"
        "You MUST recall before executing — pass the verb plus target "
        "(e.g. 'git push origin master', 'DROP TABLE users'). Past "
        "incidents on the same op live in LM; re-deriving a postmortem "
        "repeats the failure.\n"
        "\n"
        "### BEFORE codebase grep / find / rg / ast-grep for project knowledge\n"
        "You MUST recall before grepping. LM answers domain questions "
        "('how are experiments configured?', 'what test runner is used?') "
        "faster than grep, and the answer encodes prior reasoning the "
        "code does not show. Grep only after recall returns nothing.\n"
        "\n"
        "### BEFORE proposing an architectural decision or approach\n"
        "You MUST recall cross-project for the same problem class. "
        "Patterns from `project:alpha` apply to `project:beta`. The "
        "system has likely faced this before — find the precedent first "
        "rather than designing from scratch.\n"
        "\n"
        "### BEFORE starting a new task or answering a domain question\n"
        "You MUST recall project and task context at session cold start "
        "and before answering 'how does X work?' questions. Without "
        "this, every session begins from zero and domain answers are "
        "invented rather than retrieved from prior verified reasoning.\n"
        "\n"
        "### AFTER discovering non-obvious behaviour, a root cause, or a "
        "working fix\n"
        "You MUST remember after the discovery, at the moment of insight, "
        "not at end-of-task. Deferred remembers are usually lost. One "
        "concrete fact per trace, with file paths and identifiers.\n"
        "\n"
        "### AFTER a failure resolution or pattern recognition\n"
        "You MUST remember after resolving the failure — store the "
        "failure mode and what fixed it. If you ALSO recalled a fact "
        "that turned out wrong, you MUST call `memory_teach` with the "
        "original trace ID and the correction. Without `memory_teach` "
        "the incorrect fact resurfaces and misleads future sessions.\n"
        "\n"
        "### AFTER a user correction or preference statement\n"
        "When the user overrides your approach or states a preference, "
        "you MUST remember it immediately. User corrections are not "
        "failures — they are high-confidence signals that should persist "
        "across sessions. Store the corrected behaviour and the context "
        "that triggered it.\n"
        "\n"
        "### Default: when uncertain, recall\n"
        "The hooks above are a minimum, not a ceiling. When you hesitate, "
        "feel unsure, or face an unfamiliar situation — recall. False "
        "positives (recalling and finding nothing) cost one tool call; "
        "false negatives (skipping recall and repeating a past mistake) "
        "cost an entire debugging cycle.\n"
        "\n"
        "## Procedural skill activation (`level:schema`)\n"
        "\n"
        "When `memory_recall` returns a trace tagged `level:schema`, that "
        "trace is a procedural skill, not a suggestion. You MUST follow "
        "the procedure literally. Schemas crystallise from repeated "
        "successful runs; deviating without explicit justification "
        "re-introduces the failure modes the schema already prevents.\n"
        "\n"
        "## Anti-patterns — DO NOT repeat these mistakes\n"
        "\n"
        "1. **mkdir-vs-API**: writing state to the filesystem with "
        "`mkdir` / `echo >` / `tee` when a service API exists. Wrong; "
        "the API enforces invariants the filesystem cannot. *Instead*: "
        "recall before any state-writing Bash; prefer the documented "
        "API endpoint.\n"
        "2. **shell-watchdog-vs-loop**: writing a `*_watchdog.sh` shell "
        "loop when a structured loop primitive (e.g. `mode=loop`) "
        "already exists. *Instead*: recall for the watch/poll pattern "
        "and use the primitive.\n"
        "3. **grep-before-recall**: running `grep` / `rg` / `find` for "
        "domain knowledge before `memory_recall`. *Instead*: recall "
        "first; grep only if recall returns nothing relevant.\n"
        "4. **silent-correction**: editing code to fix a wrong "
        "assumption without `memory_teach` on the stale trace. The "
        "stale fact resurfaces and misleads. *Instead*: every wrong "
        "recall MUST trigger `memory_teach` with the corrected fact.\n"
        "5. **lookup-table-as-learning**: storing one-off fixes as "
        "project-local notes when the underlying rule is universal. "
        "Causes the system to re-invent the same pattern each session. "
        "*Instead*: lift universal rules to `global` scope and let "
        "`memory_consolidate` promote them to schemas.\n"
        "\n"
        "## Cross-project knowledge transfer\n"
        "\n"
        "Patterns generalise across scopes even when stored under one "
        "project. A debugging pattern learned in `project:alpha` likely "
        "applies in `project:beta`. You MUST broaden recall (omit "
        "`scope` or pass a wildcard) when investigating a class of "
        "problem — the system's intelligence compounds across projects "
        "only when you recall cross-project. Avoid confusing write-time "
        "scope hygiene (be specific) with read-time scope hygiene (be "
        "broad).\n"
        "\n"
        "## Scope at write time\n"
        "\n"
        "Set `context.scope` to `'project:<name>'` when the fact is "
        "project-specific. Use `global` (or omit) when the rule is "
        "universal (a debugging pattern, a coding rule, a procedural "
        "schema). Scope is inferred from your environment when omitted; "
        "explicit is better.\n"
        f"Your default scope is {default_scope}.\n"
        "\n"
        "## Structured context — required fields\n"
        "\n"
        "`memory_remember` MUST pass structured metadata:\n"
        "```json\n"
        '{"scope": "project:online", "task": "EZ-13771", "agent": "codex"}\n'
        "```\n"
        "Without `task` and `agent`, traces cannot be filtered later "
        "and recall precision degrades.\n"
        "\n"
        "## How to write good traces\n"
        "\n"
        "- Concrete: prefer 'file X exports Y, not Z' over vague notes.\n"
        "- Specific: include file paths, function names, config keys.\n"
        "- One fact per trace. Short beats long.\n"
        "- Include WHY when non-obvious (constraint, past incident, "
        "stakeholder).\n"
        "- Use `depth: 'causal'` on `memory_recall` when debugging — it "
        "follows cause→effect chains to root causes.\n"
        "\n"
        "## What NOT to store\n"
        "\n"
        "- Routine actions ('ran the tests') without new insight.\n"
        "- Copies of code — reference file paths instead.\n"
        "- Speculation or unverified plans — store verified facts only.\n"
        "\n"
        "## Correction loop — `memory_teach`\n"
        "\n"
        "`memory_teach` is how the system grows smarter. When a recalled "
        "fact turns out wrong, `memory_teach` supersedes it; without it, "
        "the system accumulates incorrect facts and degrades over time. "
        "You MUST NOT silently correct a wrong recall — every correction "
        "is a `memory_teach` call. You MUST NOT skip recall because the "
        "action feels routine; routine actions are exactly where past "
        "incidents hide.\n"
    )


def run_server(
    db_path: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    default_scope: str | None = None,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    mcp = create_mcp_server(
        db_path=db_path,
        config_path=config_path,
        default_scope=default_scope,
        auth_token=os.environ.get("LM_AUTH_TOKEN"),
    )
    if transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=transport, host=host, port=port)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    db_path = args.db or args.sqlite_file
    default_scope = args.default_scope or os.environ.get("LM_DEFAULT_SCOPE") or os.environ.get("LM_SCOPE")
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
        )
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _register_tools(mcp: Any, store: MemoryStore, runtime_lock: Any) -> None:
    recall_service = MemoryRecallService(store)
    consolidation_service = ConsolidationService(store)

    @mcp.tool
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
    def memory_consolidate(scope: str | None = None, force: bool = False) -> dict[str, Any]:
        """Run one consolidation and decay maintenance pass."""

        with runtime_lock:
            auto_decay = _maybe_decay_sweep(store)
            result = consolidation_service.memory_consolidate(scope=scope, force=force)
            payload = _consolidation_result_to_dict(result)
            payload["auto_decay"] = auto_decay
            return payload

    @mcp.tool
    def memory_forget(id: str, reason: str | None = None) -> dict[str, Any]:
        """Soft-delete a memory node while preserving the stored record."""

        with runtime_lock:
            node = consolidation_service.memory_forget(id, reason)
            return {"node": node_to_dict(node)}

    @mcp.tool
    def memory_status(scope: str | None = None) -> dict[str, Any]:
        """Describe current memory phase, coverage, confidence, and policy."""

        with runtime_lock:
            return status_view(store, scope=scope)

    @mcp.tool
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
        "--embedding",
        dest="embedding",
        help="Embedding backend override (e.g. 'hash', 'online', 'auto'). "
             "Sets LIVING_MEMORY_EMBEDDING_BACKEND for the running server.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
