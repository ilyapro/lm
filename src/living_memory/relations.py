"""Knowledge-relation semantics shared by runtime and offline readers.

Inferred prose relations and duplicate bookkeeping are not corrections.
Historical rows remain for inspection without validity, graph or lifecycle
authority unless they also record an explicit correction contribution.
Recall delivery history is not a relation between the displayed knowledge and
a later trace. Shared work and chronology do not establish causality.
Unrecognized metadata retains its ordinary relation semantics.
"""

from collections.abc import Mapping
import json
import sqlite3
from typing import Any

from living_memory.file_evidence import file_key, file_owner


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


def _checked_file_weight(metadata_json, weight, left_scope, left_json, right_scope, right_json, source_only):
    """Project a stored maximum onto the contributions its endpoints support."""
    metadata = json.loads(metadata_json)
    files = metadata.get("files")
    names = {f for f in files if isinstance(f, str)} if isinstance(files, list) else set()
    def citations(scope, fields):
        repo, worktree, project, paths = json.loads(fields)
        owner = file_owner(scope, repo, worktree, project)
        if not isinstance(paths, list):
            return set()
        return {(key, path) for path in paths if (key := file_key(owner, path)) is not None}

    left = citations(left_scope, left_json)
    right = citations(right_scope, right_json)
    recorded = {key for key, name in left | right if name in names}
    shared = recorded.intersection((k for k, _ in left), (k for k, _ in right))
    count = len(shared)
    if count and count == len(files) and not source_only:
        return weight
    contributions = []
    if count:
        contributions.append(shared_files_weight(shared))
    commit = metadata.get("commit")
    if (isinstance(commit, str) and len(commit) == 7) or metadata.get("kind") == "content_reference":
        contributions.append(CONTENT_REFERENCE_WEIGHT)
    if metadata.get("kind") == "derived_from":
        contributions.append(DERIVED_FROM_WEIGHT)
    basis = metadata.get("basis")
    if basis == "explicit":
        contributions.append(weight)
    if basis == "cluster" or (basis == "co_access" and not source_only):
        # A merged maximum above the recorded file contribution came from
        # another writer. Otherwise only the membership/co-access floor is known.
        floor = CLUSTER_MEMBERSHIP_WEIGHT if basis == "cluster" else 0.25
        independent = isinstance(files, list) and weight > shared_files_weight(files)
        contributions.append(weight if independent and not source_only else floor)
    return min(weight, max(contributions)) if contributions else None


def register_file_evidence(connection: sqlite3.Connection) -> None:
    connection.create_function("lm_checked_file_weight", 7, _checked_file_weight, deterministic=True)


def checked_weight_sql(alias: str = "connections", *, source_only: bool = False) -> str:
    """Only file-derived edges fetch endpoint contexts, by primary key."""
    meta = f"{alias}.metadata"
    weight = f"{alias}.weight"
    other = source_relation_weight_sql(alias) if source_only else weight
    fields = "'$.repo', '$.worktree', '$.project', '$.files'"
    return (
        f"CASE WHEN {alias}.type = 'related' AND (json_extract({meta}, '$.rule') = 'R4b'"
        f" OR json_extract({meta}, '$.basis') = 'shared_files'"
        f" OR json_extract({meta}, '$.rule_basis') = 'shared_files') THEN ("
        f"SELECT lm_checked_file_weight({meta}, {weight}, a.scope, json_extract(a.context, {fields}),"
        f" b.scope, json_extract(b.context, {fields}), {int(source_only)}) FROM nodes a JOIN nodes b"
        f" ON b.id = {alias}.target_id WHERE a.id = {alias}.source_id)"
        f" ELSE {other} END"
    )


def current_relation_sql(alias: str = "connections") -> str:
    return f"({semantic_relation_sql(alias)} AND ({checked_weight_sql(alias)}) IS NOT NULL)"


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
