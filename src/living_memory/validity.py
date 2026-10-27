"""Read supersession from relations; keep visibility separate from validity."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .scope import ScopePlan


def supersedes_sets(
    store: Any, node_ids: Iterable[str],
) -> tuple[dict[str, tuple[str, ...]], set[str]]:
    """Incident edges as (superseded -> corrections, superseding ids).

    Shared by ranking, selection and knowledge reads. Do not follow absent
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
    """Existing direct correction ids within the request's authoritative scopes.

    This batch read neither retrieves candidates nor grants feedback credit.
    Decayed nodes remain addressable history through exact lookup. Missing or
    out-of-scope corrections yield no link; their supersedes edge still counts.
    No correction body or transitive/cycle resolution enters the projection.
    """
    scopes = store.get_node_scopes({source for sources in corrections.values() for source in sources})
    return {
        target: [source for source in sources if source in scopes and plan.allows(scopes[source])]
        for target, sources in corrections.items()
    }


def validity_fields(superseded: bool, references: Sequence[str] = ()) -> dict[str, Any]:
    """Project known replacement independently of readable correction links."""
    return {"superseded": superseded, "superseded_by": list(references)}


def read_validity(
    store: Any, node_ids: Iterable[str], plan: ScopePlan,
) -> dict[str, dict[str, Any]]:
    """Batch-project incident facts for already selected nodes, without writes.

    The caller supplies the surface's existing scope admission. Recall uses
    validity_fields with its ranking observation instead of reading edges again.
    """
    ids = tuple(dict.fromkeys(node_ids))
    corrections, _ = supersedes_sets(store, ids)
    references = correction_references(
        store, {node_id: corrections.get(node_id, ()) for node_id in ids}, plan,
    )
    return {
        node_id: validity_fields(node_id in corrections, references[node_id])
        for node_id in ids
    }
