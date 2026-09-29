"""Scope resolution for hierarchical memory recall."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


GLOBAL_SCOPE = "global"
# Trailing entry of a plan that admits every scope it does not name.
ALL_SCOPES = "*"


@dataclass(frozen=True, slots=True)
class ScopePlan:
    """Search plan: ``scopes`` ordered from narrowest to broadest.

    A plan ending in :data:`ALL_SCOPES` searches the whole store; the scopes
    it names before that only rank first. Any other plan is a restriction.
    """

    requested_scope: str
    scopes: tuple[str, ...]

    @property
    def family(self) -> str:
        return scope_family(self.requested_scope)

    @property
    def restricted(self) -> bool:
        return ALL_SCOPES not in self.scopes

    @property
    def named_scopes(self) -> tuple[str, ...]:
        return tuple(scope for scope in self.scopes if scope != ALL_SCOPES)

    @property
    def search_scopes(self) -> tuple[str | None, ...]:
        """What each channel queries in turn; ``None`` is the whole store."""

        return self.scopes if self.restricted else (None,)

    @property
    def recall_scope(self) -> str | None:
        """The ``scope`` argument that re-resolves to this plan (with its ambient context)."""

        return self.requested_scope if self.restricted else None

    def allows(self, scope: str) -> bool:
        return not self.restricted or scope in self.scopes

    def rank(self, scope: str) -> int:
        """Position among the named scopes; an unnamed admitted scope ranks broadest."""

        named = self.named_scopes
        if scope in named:
            return named.index(scope)
        return len(named) - (0 if self.restricted else 1)

    def boost_steps(self, scope: str) -> int:
        """How many named scopes rank below ``scope`` (0 for the broadest)."""

        return max(0, len(self.named_scopes) - self.rank(scope) - 1)


class ScopeResolver:
    """Resolve the recall scope into a search plan.

    An explicit scope -- the argument or ``ambient_context.scope`` -- restricts
    the search to it (a session also sees its ambient project) and ``global``.
    Without one the whole store is searched: the ambient session and project,
    or the configured default project, only rank first.
    """

    def resolve(
        self,
        *,
        scope: str | None = None,
        ambient_context: Mapping[str, Any] | None = None,
        store: Any | None = None,
    ) -> ScopePlan:
        ambient = dict(ambient_context or {})
        project = _ambient_project(ambient)
        declared = scope or ambient.get("scope")
        if declared:
            return _plan_for_scope(normalize_scope(str(declared)), project)
        preferred = _ambient_session(ambient) or project
        if preferred:
            plan = _plan_for_scope(preferred, project)
        else:
            # The request stays global -- feedback closure relies on that
            # divergence from the configured default write scope -- while the
            # deployment's own project still ranks first.
            default_project = _configured_default_project(store)
            plan = ScopePlan(
                requested_scope=GLOBAL_SCOPE,
                scopes=tuple(dict.fromkeys((default_project or GLOBAL_SCOPE, GLOBAL_SCOPE))),
            )
        return replace(plan, scopes=(*plan.scopes, ALL_SCOPES))


def resolve_scope(
    *,
    scope: str | None = None,
    ambient_context: Mapping[str, Any] | None = None,
    store: Any | None = None,
) -> ScopePlan:
    return ScopeResolver().resolve(
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


def _plan_for_scope(requested: str, project_scope: str | None) -> ScopePlan:
    scopes = [requested]
    if scope_family(requested) == "session" and project_scope:
        scopes.append(project_scope)
    scopes.append(GLOBAL_SCOPE)
    return ScopePlan(requested_scope=requested, scopes=tuple(dict.fromkeys(scopes)))


def _ambient_session(ambient: Mapping[str, Any]) -> str | None:
    session_id = ambient.get("session_id") or ambient.get("session")
    return normalize_scope(f"session:{session_id}") if session_id else None


def _ambient_project(ambient: Mapping[str, Any]) -> str | None:
    workspace_path = ambient.get("workspace_path") or ambient.get("cwd")
    for raw in (
        ambient.get("project_scope"),
        ambient.get("project") or ambient.get("project_name") or ambient.get("workspace"),
        Path(str(workspace_path)).name if workspace_path else None,
    ):
        if raw:
            normalized = normalize_scope(str(raw))
            if scope_family(normalized) == "project":
                return normalized
    return None
