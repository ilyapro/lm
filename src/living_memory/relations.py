"""Knowledge-relation semantics shared by runtime and offline readers.

Inferred prose relations and duplicate bookkeeping are not corrections.
Historical rows remain for inspection without validity, graph or lifecycle
authority unless they also record an explicit correction contribution.
Recall delivery history is not a relation between the displayed knowledge and
a later trace. Shared work and chronology do not establish causality.
Unrecognized metadata retains its ordinary relation semantics.
"""

from collections.abc import Mapping
from typing import Any


_NON_CORRECTION_KINDS = ("drain_near_duplicate", "duplicate_content", "content_correction")
_HEURISTIC_PRODUCER = "lm_collapse_near_dups.py"
_HEURISTIC_REASON = "near-duplicate collapse"

# Source contributions shared with their writers. A merged row's maximum
# weight can also contain access/recall history; it is not that contribution.
CONTENT_REFERENCE_WEIGHT = 0.9
DERIVED_FROM_WEIGHT = 0.6
CLUSTER_MEMBERSHIP_WEIGHT = 0.2


def shared_files_weight(files: Any) -> float:
    return min(0.95, 0.5 + 0.15 * len(files))


def source_relation_weight(
    relation_type: str, weight: float, metadata: Mapping[str, Any],
) -> float:
    """Current source contribution without known historical reinforcement.

    Preserve independent evidence even when a merged row retains an old
    history basis. Known writer annotations recover their own contribution,
    not the row's maximum. Unknown metadata keeps its ordinary semantics.
    """
    weight = max(0.0, weight)
    if not is_semantic_relation(relation_type, metadata):
        return 0.0
    if relation_type != "related":
        return weight
    basis, kind, rule = (metadata.get(key) for key in ("basis", "kind", "rule"))
    historical = basis in ("implicit_recall_feedback", "co_access", "cluster")
    if kind == "content_reference" and (historical or rule == "R6"):
        return min(weight, CONTENT_REFERENCE_WEIGHT)
    if kind == "derived_from" and (historical or rule == "R5a"):
        return min(weight, DERIVED_FROM_WEIGHT)
    if rule == "R4a":
        return min(weight, CONTENT_REFERENCE_WEIGHT)
    if rule == "R4b" and isinstance(metadata.get("files"), list):
        return min(weight, shared_files_weight(metadata["files"]))
    if basis == "cluster":
        return min(weight, CLUSTER_MEMBERSHIP_WEIGHT)
    if basis == "co_access":
        return 0.0
    return weight


def source_relation_weight_sql(alias: str = "") -> str:
    """SQL equivalent of source_relation_weight, before adjacency limits."""
    prefix = f"{alias}." if alias else ""
    weight = f"MAX(0.0, {prefix}weight)"
    kind = f"COALESCE(json_extract({prefix}metadata, '$.kind'), '')"
    rule = f"COALESCE(json_extract({prefix}metadata, '$.rule'), '')"
    basis = f"COALESCE(json_extract({prefix}metadata, '$.basis'), '')"
    historical = f"{basis} IN ('implicit_recall_feedback', 'co_access', 'cluster')"
    return (
        f"CASE WHEN NOT {semantic_relation_sql(alias)} THEN 0.0"
        f" WHEN {prefix}type <> 'related' THEN {weight}"
        f" WHEN {kind} = 'content_reference' AND ({historical} OR {rule} = 'R6')"
        f" THEN MIN({weight}, {CONTENT_REFERENCE_WEIGHT})"
        f" WHEN {kind} = 'derived_from' AND ({historical} OR {rule} = 'R5a')"
        f" THEN MIN({weight}, {DERIVED_FROM_WEIGHT})"
        f" WHEN {rule} = 'R4a' THEN MIN({weight}, {CONTENT_REFERENCE_WEIGHT})"
        f" WHEN {rule} = 'R4b' AND json_type({prefix}metadata, '$.files') = 'array'"
        f" THEN MIN({weight}, 0.95, 0.5 + 0.15 * json_array_length({prefix}metadata, '$.files'))"
        f" WHEN {basis} = 'cluster' THEN MIN({weight}, {CLUSTER_MEMBERSHIP_WEIGHT})"
        f" WHEN {basis} = 'co_access' THEN 0.0 ELSE {weight} END"
    )


def is_correction(relation_type: str, metadata: Mapping[str, Any]) -> bool:
    if relation_type != "supersedes":
        return False
    # Teach's explicit contribution survives merged inferred metadata.
    correction = metadata.get("correction")
    if isinstance(correction, str) and correction != "":
        return True
    return not (
        metadata.get("kind") in _NON_CORRECTION_KINDS
        or metadata.get("rule") == "R1c"
        or (metadata.get("by") == _HEURISTIC_PRODUCER
            and metadata.get("reason") == _HEURISTIC_REASON)
    )


def is_semantic_relation(relation_type: str, metadata: Mapping[str, Any]) -> bool:
    if relation_type == "supersedes":
        return is_correction(relation_type, metadata)
    if relation_type == "caused":
        # A caller's explicit basis survives shallow merges of inferred markers.
        return metadata.get("basis") == "explicit" or not (
            metadata.get("rule") == "R2a"
            or metadata.get("kind") == "failure_resolution"
        )
    # Content-reference and derivation writers can merge their evidence into
    # a row without replacing an earlier recall-history basis.
    return not (
        relation_type == "related"
        and metadata.get("basis") == "implicit_recall_feedback"
        and metadata.get("kind") not in ("content_reference", "derived_from")
    )


def correction_sql(alias: str = "") -> str:
    """SQL equivalent of is_correction; alias is a caller-owned identifier."""
    prefix = f"{alias}." if alias else ""
    kinds = ", ".join(f"'{kind}'" for kind in _NON_CORRECTION_KINDS)
    return (
        f"({prefix}type = 'supersedes'"
        f" AND ((COALESCE(json_type({prefix}metadata, '$.correction'), '') = 'text'"
        f" AND json_extract({prefix}metadata, '$.correction') <> '') OR ("
        f" COALESCE(json_extract({prefix}metadata, '$.kind'), '') NOT IN ({kinds})"
        f" AND COALESCE(json_extract({prefix}metadata, '$.rule'), '') <> 'R1c'"
        f" AND NOT (COALESCE(json_extract({prefix}metadata, '$.by'), '') = '{_HEURISTIC_PRODUCER}'"
        f" AND COALESCE(json_extract({prefix}metadata, '$.reason'), '') = '{_HEURISTIC_REASON}'))))"
    )


def semantic_relation_sql(alias: str = "") -> str:
    """SQL equivalent of is_semantic_relation; filter before graph limits."""
    prefix = f"{alias}." if alias else ""
    return (
        f"(({prefix}type <> 'supersedes' OR {correction_sql(alias)})"
        f" AND ({prefix}type <> 'caused'"
        f" OR COALESCE(json_extract({prefix}metadata, '$.basis'), '') = 'explicit'"
        f" OR (COALESCE(json_extract({prefix}metadata, '$.rule'), '') <> 'R2a'"
        f" AND COALESCE(json_extract({prefix}metadata, '$.kind'), '') <> 'failure_resolution'))"
        f" AND NOT ({prefix}type = 'related' AND "
        f"COALESCE(json_extract({prefix}metadata, '$.basis'), '') = 'implicit_recall_feedback'"
        f" AND COALESCE(json_extract({prefix}metadata, '$.kind'), '')"
        f" NOT IN ('content_reference', 'derived_from')))"
    )
