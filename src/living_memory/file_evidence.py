"""Lexical file identity from recorded source fields, without filesystem access."""

from collections.abc import Iterator, Mapping
import posixpath
from typing import Any


def _path(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        return None
    if ".." in value.split("/") or value.endswith(("/", "/.")):
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
        root = roots.pop()
        if repo and not repo.startswith("/") and repo != posixpath.basename(root):
            return None
        return root or None
    if repo is not None:
        return "repo:" + repo if _path(repo) else None
    if project is not None:
        if not isinstance(project, str) or not project.strip():
            return None
        name = project.removeprefix("project:")
        return "project:" + name if name.strip() and "\0" not in name else None
    return scope if scope.startswith("project:") and scope[8:].strip() and "\0" not in scope else None


def file_key(owner: str | None, name: Any) -> str | None:
    path = _path(name)
    if path is None:
        return None
    if path.startswith("/"):
        root, marker, checkout = path.partition("/.worktrees/")
        if marker and root and "/" in checkout:
            return root + "/" + checkout.split("/", 1)[1]
        return path
    if owner is None:
        return None
    if owner.startswith("/"):
        return posixpath.join(owner, path)
    return owner + "\0" + path


def citations_for_owner(owner: str | None, files: Any) -> Iterator[tuple[str, str]]:
    """Yield (identity, recorded spelling) once per genuine file occurrence."""
    if isinstance(files, list):
        for name in files:
            key = file_key(owner, name)
            if key is not None:
                yield key, name


def file_citations(scope: str, context: Mapping[str, Any]) -> Iterator[tuple[str, str]]:
    owner = file_owner(scope, context.get("repo"), context.get("worktree"), context.get("project"))
    return citations_for_owner(owner, context.get("files"))
