"""Lexical file identity from recorded source fields, without filesystem access."""

from collections.abc import Mapping
import posixpath
from typing import Any


def _path(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        return None
    if ".." in value.split("/") or value.endswith("/"):
        return None
    path = posixpath.normpath(value)
    return path if path not in (".", "/") else None


def file_owner(scope: str, repo: Any, worktree: Any, project: Any) -> str | None:
    """Prefer a recorded repository root to a project label used for storage.

    Historical worktrees below <repo>/.worktrees/<name> carry the same
    repository identity. Two concrete roots cannot be reconciled by basename.
    Global and session scopes alone say nothing about relative paths.
    """
    roots = set()
    for value in (repo, worktree):
        if value is None:
            continue
        if not isinstance(value, str):
            return None
        if value.startswith("/"):
            root = _path(value.rstrip("/"))
            if not root:
                return None
            roots.add(root.partition("/.worktrees/")[0])
    if len(roots) > 1:
        return None
    if roots:
        return roots.pop()
    if repo is not None:
        return "repo:" + repo if _path(repo) else None
    if project is not None:
        if not isinstance(project, str) or not project.strip():
            return None
        return "project:" + project.removeprefix("project:")
    return scope if scope.startswith("project:") and scope[8:].strip() else None


def file_key(owner: str | None, name: Any) -> str | None:
    path = _path(name)
    if path is None:
        return None
    if path.startswith("/"):
        return path
    if owner is None:
        return None
    if owner.startswith("/"):
        return posixpath.join(owner, path)
    return owner + "\0" + path


def file_citations(scope: str, context: Mapping[str, Any]):
    """Yield (identity, recorded spelling) once per genuine file occurrence."""
    files = context.get("files")
    if isinstance(files, list):
        owner = file_owner(scope, context.get("repo"), context.get("worktree"), context.get("project"))
        for name in files:
            key = file_key(owner, name)
            if key is not None:
                yield key, name
