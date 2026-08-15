"""Scope resolution for hierarchical memory recall."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re


GLOBAL_SCOPE = "global"


@dataclass(frozen=True, slots=True)
class ScopePlan:
    """Resolved search plan ordered from narrowest to broadest scope."""

    requested_scope: str
    scopes: tuple[str, ...]
    explicit: bool = False
    ambient_scope: str | None = None
    implicit_scope: str | None = None

    @property
    def family(self) -> str:
        return scope_family(self.requested_scope)

    def allows(self, scope: str) -> bool:
        return scope in self.scopes

    def rank(self, scope: str) -> int:
        try:
            return self.scopes.index(scope)
        except ValueError:
            return len(self.scopes)


class ScopeResolver:
    """Resolve explicit, ambient, and query-implied scope into a safe search plan."""

    def resolve(
        self,
        *,
        query: str = "",
        scope: str | None = None,
        ambient_context: Mapping[str, Any] | None = None,
        store: Any | None = None,
    ) -> ScopePlan:
        ambient = dict(ambient_context or {})
        ambient_scope = _ambient_scope(ambient)
        ambient_project = _ambient_project(ambient, ambient_scope)

        if scope:
            requested = normalize_scope(scope)
            return _plan_for_scope(
                requested,
                explicit=True,
                ambient_scope=ambient_scope,
                project_scope=ambient_project,
            )

        if ambient_scope:
            return _plan_for_scope(
                ambient_scope,
                explicit=False,
                ambient_scope=ambient_scope,
                project_scope=ambient_project,
            )

        implicit_scope = infer_project_scope(query, store)
        if implicit_scope:
            return ScopePlan(
                requested_scope=implicit_scope,
                scopes=(implicit_scope, GLOBAL_SCOPE),
                implicit_scope=implicit_scope,
            )

        default_project = _configured_default_project(store)
        if default_project:
            # Scope-less call on a deployment whose writes default to a project
            # scope (storage honours config.default_scope on every remember).
            # The request stays global — the caller asked for nothing narrower,
            # and feedback closure relies on that divergence — but the search
            # widens to the declared project so the deployment's own memory
            # stays reachable without a deliberate query mention.
            return ScopePlan(
                requested_scope=GLOBAL_SCOPE,
                scopes=(default_project, GLOBAL_SCOPE),
            )

        return ScopePlan(requested_scope=GLOBAL_SCOPE, scopes=(GLOBAL_SCOPE,))


def resolve_scope(
    *,
    query: str = "",
    scope: str | None = None,
    ambient_context: Mapping[str, Any] | None = None,
    store: Any | None = None,
) -> ScopePlan:
    return ScopeResolver().resolve(
        query=query,
        scope=scope,
        ambient_context=ambient_context,
        store=store,
    )


def normalize_scope(scope: str) -> str:
    cleaned = str(scope).strip()
    if not cleaned:
        return GLOBAL_SCOPE
    if cleaned == GLOBAL_SCOPE:
        return GLOBAL_SCOPE
    if cleaned.startswith("project:") or cleaned.startswith("session:"):
        prefix, value = cleaned.split(":", 1)
        if not value:
            raise ValueError(f"{prefix} scope must include an id")
        return f"{prefix}:{value}"
    if ":" in cleaned:
        prefix, value = cleaned.split(":", 1)
        if prefix in {"workspace", "repo"} and value:
            return f"project:{value}"
        raise ValueError(f"unsupported scope prefix: {prefix}")
    return f"project:{cleaned}"


def scope_family(scope: str) -> str:
    if ":" in scope:
        return scope.split(":", 1)[0]
    return scope


def infer_project_scope(query: str, store: Any | None = None) -> str | None:
    direct = re.search(r"\bproject:([A-Za-z0-9_.-]+)\b", query)
    if direct:
        return f"project:{direct.group(1)}"

    if store is None:
        return None

    query_tokens = set(_tokens(query))
    if not query_tokens:
        return None

    rows = store.connection.execute(
        "SELECT DISTINCT scope FROM nodes WHERE scope LIKE 'project:%' ORDER BY scope"
    ).fetchall()
    for row in rows:
        candidate = str(row["scope"])
        project_name = candidate.split(":", 1)[1]
        if _query_mentions_project(query, query_tokens, project_name):
            return candidate
    return None


def _query_mentions_project(query: str, query_tokens: set[str], project_name: str) -> bool:
    """True only for a deliberate mention of the project.

    Either the whole name appears as a substring, or every token of the name
    appears among the query tokens. A single shared token is not a mention:
    it silently narrowed broad queries into an unrelated project's scope
    (query "pipeline docs" is not a request for project:data-pipeline).
    """

    if project_name.lower() in query.lower():
        return True
    project_tokens = set(_tokens(project_name))
    return bool(project_tokens) and project_tokens <= query_tokens


def _configured_default_project(store: Any | None) -> str | None:
    """The store's configured default write scope, when it names a project.

    Session and global defaults never widen a plan: global adds nothing, and a
    session default is not a durable declaration the way an operator-configured
    project scope is.
    """

    config = getattr(store, "config", None)
    raw = getattr(config, "default_scope", None)
    if not raw:
        return None
    try:
        normalized = normalize_scope(str(raw))
    except ValueError:
        return None
    if scope_family(normalized) != "project":
        return None
    return normalized


def _plan_for_scope(
    requested: str,
    *,
    explicit: bool,
    ambient_scope: str | None,
    project_scope: str | None,
) -> ScopePlan:
    if requested == GLOBAL_SCOPE:
        return ScopePlan(
            requested_scope=GLOBAL_SCOPE,
            scopes=(GLOBAL_SCOPE,),
            explicit=explicit,
            ambient_scope=ambient_scope,
        )

    family = scope_family(requested)
    if family == "project":
        return ScopePlan(
            requested_scope=requested,
            scopes=(requested, GLOBAL_SCOPE),
            explicit=explicit,
            ambient_scope=ambient_scope,
        )

    if family == "session":
        scopes = [requested]
        if project_scope and project_scope != requested:
            scopes.append(project_scope)
        scopes.append(GLOBAL_SCOPE)
        return ScopePlan(
            requested_scope=requested,
            scopes=tuple(dict.fromkeys(scopes)),
            explicit=explicit,
            ambient_scope=ambient_scope,
        )

    raise ValueError(f"unsupported scope family: {family}")


def _ambient_scope(ambient: Mapping[str, Any]) -> str | None:
    raw_scope = ambient.get("scope")
    if raw_scope:
        return normalize_scope(str(raw_scope))

    session_id = ambient.get("session_id") or ambient.get("session")
    if session_id:
        return normalize_scope(f"session:{session_id}")

    project = ambient.get("project") or ambient.get("project_name") or ambient.get("workspace")
    if project:
        return normalize_scope(str(project))

    workspace_path = ambient.get("workspace_path") or ambient.get("cwd")
    if workspace_path:
        name = Path(str(workspace_path)).name
        if name:
            return normalize_scope(name)

    return None


def _ambient_project(ambient: Mapping[str, Any], ambient_scope: str | None) -> str | None:
    if ambient_scope and scope_family(ambient_scope) == "project":
        return ambient_scope

    project_scope = ambient.get("project_scope")
    if project_scope:
        normalized = normalize_scope(str(project_scope))
        if scope_family(normalized) == "project":
            return normalized

    project = ambient.get("project") or ambient.get("project_name") or ambient.get("workspace")
    if project:
        normalized = normalize_scope(str(project))
        if scope_family(normalized) == "project":
            return normalized

    workspace_path = ambient.get("workspace_path") or ambient.get("cwd")
    if workspace_path:
        name = Path(str(workspace_path)).name
        if name:
            return normalize_scope(name)

    return None


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+", value.lower())
