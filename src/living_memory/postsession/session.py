"""Stable session schema for Living Memory's offline post-session extraction.

The shape of the stage is::

    transcript file --(transcripts.py adapter)--> SessionRecord
    SessionRecord   --(insights / corrections / evidence)--> ProposedOp

A :class:`SessionRecord` is the *reconstruction* of one finished agent
session. It deliberately keeps the exact identifiers the live server handed
back during the session, because those identifiers -- and not any heuristic
re-matching -- are what make the extraction stage exact:

``RecallInteraction.recall_event_id``
    the ``recall_events`` row the server created for that recall; the key a
    retroactive grounded-usage attestation has to credit.
``RecallInteraction.delivered[].node_id``
    exactly which memory nodes the agent was shown, with the per-channel
    scores that put them there.
``WriteInteraction.node_id``
    the node the session actually created, plus the ``prior_recalls`` and
    ``implicit_feedback`` the server attached to it.
``FileMutation.unified_diff`` / ``pre_image``
    the observable reality a recalled fact can be checked against.

Published contract
------------------
Every downstream child of the post-session stage imports these types, so they
are a *published contract*:

* a field may be **added** -- it must carry a default, and a consumer written
  against the older shape keeps working because :meth:`from_dict` ignores keys
  it does not know;
* a field name, its meaning, or its wire representation may **not** change,
  and a field may not be removed, without bumping :data:`SCHEMA_VERSION` and
  updating every consumer in the same change.

``tests/test_postsession_session.py`` pins the published field name set of
every class: it fails on a rename or a removal and stays quiet on an addition,
which is exactly the asymmetry above.

Serialization
-------------
``to_dict``/``from_dict`` are generic and driven by :func:`dataclasses.fields`
plus the resolved type hints (see :class:`Wire`). This is deliberate: the
previous draft of this module hand-wrote nine ``to_dict`` and four
``from_dict`` methods, ~166 lines whose only failure mode was silent -- adding
a field to a dataclass and forgetting one of its two hand-written halves lost
that field on the wire with no error anywhere. Field-driven serialization
cannot drift, so "fields may be added" is safe to promise.

The wire form is plain JSON: tuples become lists, nested dataclasses become
dicts, and ``from_dict`` restores the declared types (including tuple-ness) so
that ``T.from_dict(x.to_dict()) == x`` for every type here.

Content policy: a ``SessionRecord`` lives in memory and *does* carry transcript
content, because the extractor has to read it. Nothing in this module writes
to disk, reads the Living Memory database, or calls the server. The tracked
corpus manifest (``corpus.py``) carries no content at all.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
from types import UnionType
from typing import Any, Literal, Self, Union, get_args, get_origin, get_type_hints

SCHEMA_VERSION = 1

#: Transcript sources understood by :mod:`living_memory.postsession.transcripts`.
SOURCES: tuple[str, ...] = (
    "claude",
    "codex",
    "ae_chat",
    "ae_node_result",
    "gigacode",
    "deepseek",
)

#: CLI that produced a session. ``ae`` marks a dashboard chat whose underlying
#: CLI can change per turn; the per-turn CLI is then on the turn itself.
CLIS: tuple[str, ...] = ("claude", "codex", "gigacode", "deepseek", "ae", "unknown")

#: The three write kinds a downstream child may propose.
OP_KINDS: tuple[str, ...] = ("remember", "teach", "attest")

#: Living Memory MCP tool names, however the host CLI spells the prefix.
MEMORY_SERVER = "living-memory"
RECALL_TOOL = "memory_recall"
REMEMBER_TOOL = "memory_remember"
TEACH_TOOL = "memory_teach"

MutationKind = Literal["create", "update", "delete", "unknown"]
OpKind = Literal["remember", "teach", "attest"]

_NONE_TYPE = type(None)
_SCALARS: tuple[type, ...] = (str, int, float, bool)


def normalize_tool_name(name: str | None) -> tuple[str | None, str]:
    """Split a host-specific tool name into ``(server, tool)``.

    The same MCP tool is spelled three ways across the corpus::

        claude    mcp__living-memory__memory_recall
        ae_chat   living-memory.memory_recall
        codex     server="living-memory", tool="memory_recall"  (already split)

    A plain built-in tool (``Bash``, ``edit_file``) yields ``(None, name)``.
    """

    raw = (name or "").strip()
    if not raw:
        return None, ""
    if raw.startswith("mcp__"):
        parts = raw[len("mcp__") :].split("__", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], ""
    if "." in raw and " " not in raw:
        server, _, tool = raw.partition(".")
        if server and tool:
            return server, tool
    return None, raw


# --------------------------------------------------------------------------
# generic dataclass <-> JSON serialization
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _hints(cls: type) -> Mapping[str, Any]:
    """Resolved type hints for ``cls``.

    ``from __future__ import annotations`` makes every annotation a string, so
    the hints have to be resolved once against the defining module. Cached
    because ``from_dict`` is on the hot path of loading a whole corpus.
    """

    return get_type_hints(cls)


def _to_wire(value: Any) -> Any:
    """Render a runtime value as JSON-clean data (tuples become lists)."""

    if isinstance(value, Wire):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_wire(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _to_wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wire(item) for item in value]
    return value


def _from_wire(hint: Any, value: Any) -> Any:
    """Rebuild a value of declared type ``hint`` from its wire form.

    Unions are followed through their single non-``None`` member (``str |
    None``); an ambiguous union (``int | str``) and ``Any`` pass through
    untouched, which is what ``RecallInteraction.depth`` needs.
    """

    origin = get_origin(hint)
    if origin is Union or origin is UnionType:
        if value is None:
            return None
        members = [arg for arg in get_args(hint) if arg is not _NONE_TYPE]
        return _from_wire(members[0], value) if len(members) == 1 else value
    if value is None:
        return None
    if origin is list:
        args = get_args(hint) or (Any,)
        return [_from_wire(args[0], item) for item in value]
    if origin is tuple:
        args = get_args(hint)
        if not args:
            return tuple(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_from_wire(args[0], item) for item in value)
        return tuple(_from_wire(arg, item) for arg, item in zip(args, value))
    if origin is dict:
        args = get_args(hint)
        item_hint = args[1] if len(args) == 2 else Any
        return {str(key): _from_wire(item_hint, item) for key, item in value.items()}
    if origin is Literal:
        return value
    if isinstance(hint, type) and issubclass(hint, Wire):
        return hint.from_dict(value)
    if hint in _SCALARS:
        return hint(value)
    return value


class Wire:
    """Field-driven JSON serialization for the schema dataclasses.

    Mixing this in gives a dataclass ``to_dict``/``from_dict`` derived from
    :func:`dataclasses.fields`, so a new field is carried by both halves the
    moment it is declared. Subclasses must not hand-write either method; the
    one sanctioned override is stamping extra constants onto the wire form
    (see :meth:`SessionRecord.to_dict`), which is field-independent.
    """

    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        """Wire form: one key per declared field, in declaration order."""

        return {f.name: _to_wire(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Rebuild from a wire dict.

        Keys the class does not declare are ignored (a newer producer may have
        added fields); declared keys that are absent fall back to the field's
        default, and a *required* field that is absent raises ``TypeError``
        from the constructor rather than being silently invented.
        """

        hints = _hints(cls)
        return cls(
            **{
                f.name: _from_wire(hints[f.name], data[f.name])
                for f in fields(cls)  # type: ignore[arg-type]
                if f.init and f.name in data
            }
        )


def published_fields(cls: type[Wire]) -> tuple[str, ...]:
    """Field names ``cls`` puts on the wire, in order -- the frozen contract."""

    return tuple(f.name for f in fields(cls))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceSpan(Wire):
    """Where in a transcript a fact came from.

    ``record_index`` is the 0-based ordinal of the record inside the transcript
    as the adapter streamed it (JSONL line ordinal, JSON array element ordinal,
    or text line ordinal for ``ae_node_result``). ``field`` is an optional
    dotted path inside that record.
    """

    source: str
    path: str
    record_index: int
    field: str = ""

    def locator(self) -> str:
        """Render a stable one-line locator, e.g. ``claude:/p/a.jsonl#42:mcpMeta``."""

        base = f"{self.source}:{self.path}#{self.record_index}"
        return f"{base}:{self.field}" if self.field else base

    @classmethod
    def parse(cls, locator: str) -> "SourceSpan":
        head, _, tail = locator.rpartition("#")
        source, _, path = head.partition(":")
        index, _, field_path = tail.partition(":")
        return cls(
            source=source, path=path, record_index=int(index or 0), field=field_path
        )


@dataclass(slots=True)
class Turn(Wire):
    """One conversational turn (user prompt, assistant message, system note)."""

    index: int
    role: str
    text: str
    at: str | None
    span: SourceSpan
    cli: str | None = None
    truncated: bool = False


@dataclass(slots=True)
class ToolCall(Wire):
    """A tool invocation recovered from the transcript.

    ``arguments`` is the *call* payload; ``result_text`` is the raw textual
    result when the host recorded one. Memory-tool calls are additionally
    projected into :class:`RecallInteraction` / :class:`WriteInteraction`, which
    is what downstream children should read -- ``ToolCall`` keeps the generic
    view (shell commands, file reads) used for evidence quoting.
    """

    ordinal: int
    name: str
    server: str | None
    arguments: dict[str, Any]
    span: SourceSpan
    call_id: str | None = None
    at: str | None = None
    ok: bool | None = None
    result_text: str | None = None
    result_truncated: bool = False

    @property
    def is_memory_tool(self) -> bool:
        return self.server == MEMORY_SERVER


@dataclass(slots=True)
class FileMutation(Wire):
    """A file the session actually changed.

    ``unified_diff`` is normalised across hosts: Claude's ``structuredPatch``
    hunk arrays are rendered back into unified-diff text, codex's
    ``patch_apply_end`` already ships ``unified_diff``. ``pre_image`` is the
    file content *before* the change when the host recorded it (Claude
    ``originalFile``); ``post_image`` the content after, for whole-file writes.
    """

    ordinal: int
    path: str
    kind: MutationKind
    span: SourceSpan
    unified_diff: str | None = None
    pre_image: str | None = None
    post_image: str | None = None
    at: str | None = None
    tool: str = ""
    truncated: bool = False


@dataclass(slots=True)
class DeliveredNode(Wire):
    """One memory node the server actually delivered into the session."""

    node_id: str
    rank: int
    content: str | None = None
    scope: str | None = None
    level: str | None = None
    agent: str | None = None
    task: str | None = None
    created_at: str | None = None
    score: float | None = None
    bm25_score: float | None = None
    vector_score: float | None = None
    graph_score: float | None = None
    trigger_score: float | None = None
    methods: tuple[str, ...] = ()
    delivery: str | None = None
    content_truncated: bool = False


@dataclass(slots=True)
class RecallInteraction(Wire):
    """One ``memory_recall`` round trip, with the server's own event id.

    ``ordinal`` counts recalls within the session (0-based). ``position`` is
    the fractional position of the *call* in the transcript --
    ``record_index / max(1, last_record_index)`` -- which is the same quantity
    the field measurement used to show that recall is a start-of-session ritual
    (median 0.02) while remember is a closing one (median 0.96).
    """

    ordinal: int
    query: str
    span: SourceSpan
    recall_event_id: str | None = None
    scope: str | None = None
    depth: Any = None
    max_results: int | None = None
    ambient_context: dict[str, Any] = field(default_factory=dict)
    delivered: list[DeliveredNode] = field(default_factory=list)
    at: str | None = None
    turn_index: int = -1
    record_index: int = -1
    position: float = 0.0
    truncated: bool = False

    @property
    def delivered_node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.delivered)


@dataclass(slots=True)
class WriteInteraction(Wire):
    """One ``memory_remember`` / ``memory_teach`` round trip.

    ``implicit_feedback`` is the server's own verdict block
    (``recall_event_ids``, ``linked_node_ids``, ``feedback_applied``): it says
    which recalls the write closed the loop on, and whether credit was applied.
    ``supersedes`` is set for teach only and holds the corrected node id.
    """

    ordinal: int
    kind: str
    span: SourceSpan
    node_id: str | None = None
    content: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    implicit_feedback: dict[str, Any] = field(default_factory=dict)
    prior_recall_ids: tuple[str, ...] = ()
    supersedes: str | None = None
    transport_session_id: str | None = None
    at: str | None = None
    turn_index: int = -1
    record_index: int = -1
    position: float = 0.0
    truncated: bool = False

    @property
    def feedback_applied(self) -> bool:
        return bool(self.implicit_feedback.get("feedback_applied"))


@dataclass(slots=True)
class SessionRecord(Wire):
    """One finished agent session, reconstructed from its transcript."""

    session_key: str
    source: str
    cli: str
    path: str
    cli_session_id: str | None = None
    model: str | None = None
    cwd: str | None = None
    repo: str | None = None
    git_branch: str | None = None
    git_commit: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    transport_session_ids: tuple[str, ...] = ()
    linked_session_ids: tuple[str, ...] = ()
    transcript_sha256: str | None = None
    record_count: int = 0
    turns: list[Turn] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    file_mutations: list[FileMutation] = field(default_factory=list)
    recalls: list[RecallInteraction] = field(default_factory=list)
    writes: list[WriteInteraction] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)

    @property
    def recall_event_ids(self) -> tuple[str, ...]:
        return tuple(r.recall_event_id for r in self.recalls if r.recall_event_id)

    @property
    def delivered_node_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for recall in self.recalls:
            for node in recall.delivered:
                seen.setdefault(node.node_id, None)
        return tuple(seen)

    @property
    def written_node_ids(self) -> tuple[str, ...]:
        return tuple(w.node_id for w in self.writes if w.node_id)

    @property
    def mutated_paths(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for mutation in self.file_mutations:
            seen.setdefault(mutation.path, None)
        return tuple(seen)

    def span(self, record_index: int, field_path: str = "") -> SourceSpan:
        return SourceSpan(self.source, self.path, record_index, field_path)

    def summary(self) -> dict[str, Any]:
        """Counts only -- safe to log or persist next to a tracked artifact."""

        return {
            "session_key": self.session_key,
            "source": self.source,
            "cli": self.cli,
            "records": self.record_count,
            "turns": len(self.turns),
            "tool_calls": len(self.tool_calls),
            "file_mutations": len(self.file_mutations),
            "recalls": len(self.recalls),
            "recall_event_ids": len(self.recall_event_ids),
            "delivered_nodes": len(self.delivered_node_ids),
            "writes": len(self.writes),
            "written_nodes": len(self.written_node_ids),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    def to_dict(self) -> dict[str, Any]:
        """Field-driven wire form, stamped with the schema version.

        The stamp is the one sanctioned reason to override: it adds a constant,
        never a field, so this method cannot drift when a field is added.

        ``Wire.to_dict(self)`` rather than ``super().to_dict()``: ``@dataclass(
        slots=True)`` rebuilds the class, and the zero-argument ``super()`` of
        a method defined on the *original* class then raises ``TypeError``.
        """

        return {"schema_version": SCHEMA_VERSION, **Wire.to_dict(self)}


@dataclass(frozen=True, slots=True)
class Evidence(Wire):
    """A verbatim quote from the session plus where it was said."""

    quote: str
    locator: str


@dataclass(frozen=True, slots=True)
class Provenance(Wire):
    """Which session, which exact bytes, which span produced an op."""

    session_key: str
    transcript_sha256: str
    source_span: str


class ProposedOpError(ValueError):
    """A proposed op violated the shared schema."""


@dataclass(frozen=True, slots=True)
class ProposedOp(Wire):
    """The one shape every downstream child of this stage emits.

    Nothing here writes to Living Memory: an op is a *proposal* that the
    runner gates, and only then executes over MCP.

    ``payload`` is kind-specific and is passed through untouched:

    ``remember``
        ``{"content": str, "context": {...}}`` -- the ``memory_remember`` call.
    ``teach``
        ``{"trace_id": str, "correction": str | dict, "confidence": float|None,
        "context": {...}}`` -- the ``memory_teach`` call, whose ``trace_id``
        must name a node the session was actually shown.
    ``attest``
        ``{"recall_event_id": str, "evidence": [...]}`` -- retroactive grounded
        credit for a recall, recomputed server-side from the evidence.

    :meth:`from_dict` is deliberately permissive so that a malformed op can be
    loaded and *then* rejected with a specific message; :meth:`validate` is the
    only gate, and every producer must call it before the op leaves the child.
    """

    kind: OpKind
    payload: dict[str, Any]
    provenance: Provenance
    evidence: tuple[Evidence, ...] = ()

    def validate(self) -> "ProposedOp":
        """Raise :class:`ProposedOpError` unless the op is structurally sound."""

        if self.kind not in OP_KINDS:
            raise ProposedOpError(f"unknown op kind: {self.kind!r}")
        if not isinstance(self.payload, dict) or not self.payload:
            raise ProposedOpError(f"{self.kind}: payload must be a non-empty dict")
        if not self.provenance.session_key:
            raise ProposedOpError(f"{self.kind}: provenance.session_key is required")
        if not self.provenance.transcript_sha256:
            raise ProposedOpError(
                f"{self.kind}: provenance.transcript_sha256 is required"
            )
        if not self.provenance.source_span:
            raise ProposedOpError(f"{self.kind}: provenance.source_span is required")
        if not self.evidence:
            raise ProposedOpError(f"{self.kind}: at least one evidence item is required")
        for item in self.evidence:
            if not item.quote.strip():
                raise ProposedOpError(f"{self.kind}: evidence quote must be non-empty")
            if not item.locator.strip():
                raise ProposedOpError(f"{self.kind}: evidence locator must be non-empty")
        return self

    @classmethod
    def for_session(
        cls,
        kind: OpKind,
        payload: dict[str, Any],
        record: SessionRecord,
        span: SourceSpan,
        evidence: Iterable[Evidence],
    ) -> "ProposedOp":
        """Build an op whose provenance is pinned to ``record``'s exact bytes."""

        return cls(
            kind=kind,
            payload=payload,
            provenance=Provenance(
                session_key=record.session_key,
                transcript_sha256=record.transcript_sha256 or "",
                source_span=span.locator(),
            ),
            evidence=tuple(evidence),
        )


#: Every wire type this module publishes, in dependency order. Tests iterate
#: this so a new schema dataclass is covered by the contract checks the moment
#: it is added here.
SCHEMA_TYPES: tuple[type[Wire], ...] = (
    SourceSpan,
    Turn,
    ToolCall,
    FileMutation,
    DeliveredNode,
    RecallInteraction,
    WriteInteraction,
    SessionRecord,
    Evidence,
    Provenance,
    ProposedOp,
)
