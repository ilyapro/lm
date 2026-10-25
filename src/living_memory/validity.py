"""Read supersession from relations; keep visibility separate from validity."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .scope import ScopePlan


def supersedes_sets(
    store: Any, node_ids: Iterable[str],
) -> tuple[dict[str, tuple[str, ...]], set[str]]:
    """Incident edges as (superseded -> corrections, superseding ids).

    Shared by ranking, selection and exact-id reads. Do not follow absent
    intermediates or discard edges whose correction cannot be delivered.
    Runtime storage and replay expose the same incident-pair reader.
    """
    corrections: dict[str, set[str]] = {}
    superseding: set[str] = set()
    for source, target in store.list_supersedes_pairs(node_ids):
        superseding.add(source)
        corrections.setdefault(target, set()).add(source)
    return (
        {target: tuple(sorted(sources)) for target, sources in corrections.items()},
        superseding,
    )


def correction_references(
    store: Any, corrections: Mapping[str, Sequence[str]], plan: ScopePlan,
) -> dict[str, list[str]]:
    """Direct correction ids readable by lookup within the request's scopes.

    This batch read neither retrieves candidates nor grants feedback credit.
    Decayed nodes remain addressable history through exact lookup. Missing or
    out-of-scope corrections yield no link; their supersedes edge still counts.
    No correction body or transitive/cycle resolution enters the projection.
    """
    nodes = store.get_nodes({source for sources in corrections.values() for source in sources})
    return {
        target: [source for source in sources if source in nodes and plan.allows(nodes[source].scope)]
        for target, sources in corrections.items()
    }
