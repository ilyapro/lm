"""Contract tests for the published post-session session schema.

Three things are checked here, and each one guards a specific way the schema
could rot:

* ``ProposedOp.validate`` rejects every malformed shape -- an op that reaches
  the runner without provenance or evidence cannot be audited back to the
  session it claims to come from, so validation is the gate, not a courtesy;
* every dataclass round-trips through ``to_dict``/``from_dict`` *without*
  hand-written per-field code, including a dataclass this module invents that
  ``session.py`` has never seen -- that is what makes "fields may be added"
  a safe promise;
* the published field names are pinned, so a rename or a removal fails while
  an addition stays quiet.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
import json

import pytest

from living_memory.postsession.session import (
    OP_KINDS,
    SCHEMA_TYPES,
    SCHEMA_VERSION,
    DeliveredNode,
    Evidence,
    FileMutation,
    ProposedOp,
    ProposedOpError,
    Provenance,
    RecallInteraction,
    SessionRecord,
    SourceSpan,
    ToolCall,
    Turn,
    Wire,
    WriteInteraction,
    normalize_tool_name,
    published_fields,
)

# --------------------------------------------------------------------------
# fixtures -- structure only, no real transcript content
# --------------------------------------------------------------------------

SESSION_KEY = "claude:11111111-1111-4111-8111-111111111111"
TRANSCRIPT_SHA = "a" * 64


def make_record() -> SessionRecord:
    """A session record with every field and every list populated."""

    path = "/home/user/.claude/projects/-home-user-p-demo/s.jsonl"
    span = SourceSpan("claude", path, 3, "mcpMeta.structuredContent")
    return SessionRecord(
        session_key=SESSION_KEY,
        source="claude",
        cli="claude",
        path=path,
        cli_session_id="11111111-1111-4111-8111-111111111111",
        model="claude-opus-5",
        cwd="/home/user/p/demo",
        repo="/home/user/p/demo",
        git_branch="master",
        git_commit="0" * 40,
        started_at="2026-08-19T09:00:00Z",
        ended_at="2026-08-19T09:40:00Z",
        transport_session_ids=("t-1", "t-2"),
        linked_session_ids=("22222222-2222-4222-8222-222222222222",),
        transcript_sha256=TRANSCRIPT_SHA,
        record_count=17,
        turns=[
            Turn(0, "user", "REDACTED-PROMPT", "2026-08-19T09:00:01Z", span),
            Turn(
                1,
                "assistant",
                "REDACTED-ANSWER",
                None,
                span,
                cli="claude",
                truncated=True,
            ),
        ],
        tool_calls=[
            ToolCall(
                ordinal=0,
                name="memory_recall",
                server="living-memory",
                arguments={"query": "REDACTED-QUERY", "depth": "causal"},
                span=span,
                call_id="toolu_1",
                at="2026-08-19T09:00:02Z",
                ok=True,
                result_text="REDACTED-RESULT",
            ),
            ToolCall(1, "Bash", None, {"command": "pytest -q"}, span, ok=False),
        ],
        file_mutations=[
            FileMutation(
                ordinal=0,
                path="src/living_memory/postsession/session.py",
                kind="update",
                span=span,
                unified_diff="@@ -1 +1 @@\n-a\n+b\n",
                pre_image="a\n",
                post_image="b\n",
                at="2026-08-19T09:10:00Z",
                tool="Edit",
            ),
            FileMutation(1, "notes.md", "create", span, post_image="x\n", tool="Write"),
        ],
        recalls=[
            RecallInteraction(
                ordinal=0,
                query="REDACTED-QUERY",
                span=span,
                recall_event_id="01M0BNNKQCYKSH4V4M8QRQ5MPY",
                scope="project:lm",
                depth="causal",
                max_results=5,
                ambient_context={"task": "postsession"},
                delivered=[
                    DeliveredNode(
                        node_id="01M0BKS48MVR2XHNKJRB2QF7M8",
                        rank=0,
                        content="REDACTED-NODE",
                        scope="project:lm",
                        level="trace",
                        agent="claude",
                        task="postsession",
                        created_at="2026-08-18T23:37:20Z",
                        score=0.61695,
                        bm25_score=1.0,
                        vector_score=0.468668,
                        graph_score=0.0,
                        trigger_score=0.0,
                        methods=("bm25", "vector"),
                        delivery="full",
                    ),
                    DeliveredNode("01KV47B1HCYEY17D1E5RV6ZS3W", 1, content_truncated=True),
                ],
                at="2026-08-19T09:00:03Z",
                turn_index=0,
                record_index=3,
                position=0.02,
            )
        ],
        writes=[
            WriteInteraction(
                ordinal=0,
                kind="remember",
                span=span,
                node_id="01M0C000000000000000000000",
                content="REDACTED-FACT",
                context={"scope": "project:lm"},
                implicit_feedback={
                    "recall_event_ids": ["01M0BNNKQCYKSH4V4M8QRQ5MPY"],
                    "linked_node_ids": ["01M0BKS48MVR2XHNKJRB2QF7M8"],
                    "feedback_applied": True,
                },
                prior_recall_ids=("01M0BNNKQCYKSH4V4M8QRQ5MPY",),
                transport_session_id="t-1",
                at="2026-08-19T09:39:00Z",
                turn_index=1,
                record_index=16,
                position=0.96,
            ),
            WriteInteraction(
                1,
                "teach",
                span,
                node_id="01M0C111111111111111111111",
                supersedes="01KV47B1HCYEY17D1E5RV6ZS3W",
            ),
        ],
        parse_warnings=["1 record skipped"],
    )


def make_op() -> ProposedOp:
    record = make_record()
    return ProposedOp.for_session(
        "remember",
        {"content": "REDACTED-FACT", "context": {"scope": "project:lm"}},
        record,
        record.recalls[0].span,
        [Evidence("REDACTED-QUOTE", record.recalls[0].span.locator())],
    )


# --------------------------------------------------------------------------
# ProposedOp validation
# --------------------------------------------------------------------------


def test_valid_op_validates_and_pins_provenance() -> None:
    op = make_op().validate()
    assert op.kind in OP_KINDS
    assert op.provenance.session_key == SESSION_KEY
    assert op.provenance.transcript_sha256 == TRANSCRIPT_SHA
    assert op.provenance.source_span.startswith("claude:")
    assert op.evidence[0].locator == op.provenance.source_span


def test_for_session_tolerates_an_unhashed_record() -> None:
    record = make_record()
    record.transcript_sha256 = None
    op = ProposedOp.for_session(
        "attest", {"recall_event_id": "e"}, record, record.span(0), [Evidence("q", "l")]
    )
    assert op.provenance.transcript_sha256 == ""
    with pytest.raises(ProposedOpError, match="transcript_sha256"):
        op.validate()


@pytest.mark.parametrize("kind", OP_KINDS)
def test_every_declared_kind_validates(kind: str) -> None:
    op = make_op()
    assert ProposedOp(kind, op.payload, op.provenance, op.evidence).validate()  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        pytest.param(lambda op: ProposedOp("bogus", op.payload, op.provenance, op.evidence), "unknown op kind", id="unknown-kind"),  # type: ignore[arg-type]
        pytest.param(lambda op: ProposedOp("", op.payload, op.provenance, op.evidence), "unknown op kind", id="empty-kind"),  # type: ignore[arg-type]
        pytest.param(lambda op: ProposedOp("remember", {}, op.provenance, op.evidence), "non-empty dict", id="empty-payload"),
        pytest.param(lambda op: ProposedOp("remember", ["content"], op.provenance, op.evidence), "non-empty dict", id="non-dict-payload"),  # type: ignore[arg-type]
        pytest.param(lambda op: ProposedOp("remember", op.payload, Provenance("", TRANSCRIPT_SHA, "s"), op.evidence), "session_key", id="no-session-key"),
        pytest.param(lambda op: ProposedOp("remember", op.payload, Provenance(SESSION_KEY, "", "s"), op.evidence), "transcript_sha256", id="no-sha"),
        pytest.param(lambda op: ProposedOp("remember", op.payload, Provenance(SESSION_KEY, TRANSCRIPT_SHA, ""), op.evidence), "source_span", id="no-span"),
        pytest.param(lambda op: ProposedOp("remember", op.payload, op.provenance, ()), "at least one evidence", id="no-evidence"),
        pytest.param(lambda op: ProposedOp("remember", op.payload, op.provenance, (Evidence("  \n", "l"),)), "evidence quote", id="blank-quote"),
        pytest.param(lambda op: ProposedOp("remember", op.payload, op.provenance, (Evidence("q", " "),)), "evidence locator", id="blank-locator"),
        pytest.param(lambda op: ProposedOp("remember", op.payload, op.provenance, (op.evidence[0], Evidence("q", ""))), "evidence locator", id="second-item-bad"),
    ],
)
def test_validate_rejects_malformed_ops(broken, message: str) -> None:
    with pytest.raises(ProposedOpError, match=message):
        broken(make_op()).validate()


def test_from_dict_is_permissive_and_validate_is_the_gate() -> None:
    """A malformed op must be loadable, so it can be rejected with a reason."""

    wire = {
        "kind": "remember",
        "payload": {},
        "provenance": {"session_key": "", "transcript_sha256": "", "source_span": ""},
        "evidence": [],
    }
    op = ProposedOp.from_dict(wire)
    assert op.provenance == Provenance("", "", "")
    assert op.evidence == ()
    with pytest.raises(ProposedOpError):
        op.validate()


# --------------------------------------------------------------------------
# round-tripping
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", SCHEMA_TYPES, ids=lambda cls: cls.__name__)
def test_wire_form_has_exactly_the_declared_fields(cls: type[Wire]) -> None:
    """to_dict is field-driven: no field is dropped, none is invented."""

    record = make_record()
    sample = {
        SourceSpan: record.recalls[0].span,
        Turn: record.turns[1],
        ToolCall: record.tool_calls[0],
        FileMutation: record.file_mutations[0],
        DeliveredNode: record.recalls[0].delivered[0],
        RecallInteraction: record.recalls[0],
        WriteInteraction: record.writes[0],
        SessionRecord: record,
        Evidence: Evidence("q", "l"),
        Provenance: make_op().provenance,
        ProposedOp: make_op(),
    }[cls]
    expected = set(published_fields(cls))
    if cls is SessionRecord:
        expected |= {"schema_version"}
    assert set(sample.to_dict()) == expected


def test_session_record_round_trips_through_json() -> None:
    record = make_record()
    wire = record.to_dict()

    assert wire["schema_version"] == SCHEMA_VERSION
    assert json.loads(json.dumps(wire)) == wire, "wire form must be plain JSON"

    restored = SessionRecord.from_dict(json.loads(json.dumps(wire)))
    assert restored == record
    assert restored.to_dict() == wire


def test_from_dict_restores_declared_container_types() -> None:
    """Tuple-ness and nested dataclasses survive the JSON round trip."""

    restored = SessionRecord.from_dict(json.loads(json.dumps(make_record().to_dict())))

    assert isinstance(restored.transport_session_ids, tuple)
    assert isinstance(restored.linked_session_ids, tuple)
    assert isinstance(restored.turns[0], Turn)
    assert isinstance(restored.turns[0].span, SourceSpan)
    assert isinstance(restored.recalls[0].delivered[0], DeliveredNode)
    assert isinstance(restored.recalls[0].delivered[0].methods, tuple)
    assert isinstance(restored.writes[0].prior_recall_ids, tuple)
    assert isinstance(restored.file_mutations[0], FileMutation)
    assert isinstance(restored.tool_calls[0].arguments, dict)
    assert restored.recalls[0].depth == "causal", "Any-typed fields pass through"


@pytest.mark.parametrize(
    "sample",
    [
        SourceSpan("claude", "/tmp/a.jsonl", 42, "mcpMeta"),
        SourceSpan("codex", "/tmp/b.jsonl", 0),
        Evidence("REDACTED-QUOTE", "claude:/tmp/a.jsonl#42"),
        Provenance(SESSION_KEY, TRANSCRIPT_SHA, "claude:/tmp/a.jsonl#42"),
    ],
    ids=lambda s: type(s).__name__,
)
def test_small_value_types_round_trip(sample: Wire) -> None:
    assert type(sample).from_dict(sample.to_dict()) == sample


def test_proposed_op_round_trips() -> None:
    op = make_op().validate()
    assert ProposedOp.from_dict(op.to_dict()) == op
    assert ProposedOp.from_dict(json.loads(json.dumps(op.to_dict()))) == op
    assert isinstance(ProposedOp.from_dict(op.to_dict()).evidence, tuple)
    assert isinstance(ProposedOp.from_dict(op.to_dict()).provenance, Provenance)


def test_source_span_locator_round_trips() -> None:
    span = SourceSpan("claude", "/tmp/a.jsonl", 42, "mcpMeta.structuredContent")
    assert span.locator() == "claude:/tmp/a.jsonl#42:mcpMeta.structuredContent"
    assert SourceSpan.parse(span.locator()) == span
    bare = SourceSpan("codex", "/tmp/b.jsonl", 7)
    assert bare.locator() == "codex:/tmp/b.jsonl#7"
    assert SourceSpan.parse(bare.locator()) == bare


def test_from_dict_ignores_unknown_keys() -> None:
    """A newer producer may add fields; an older consumer must not break."""

    wire = make_record().recalls[0].span.to_dict() | {"future_field": "x"}
    assert SourceSpan.from_dict(wire) == make_record().recalls[0].span


def test_from_dict_applies_declared_defaults() -> None:
    turn = Turn.from_dict({"index": 0, "role": "user", "text": "t", "at": None, "span": {"source": "claude", "path": "/p", "record_index": 0}})
    assert turn.cli is None and turn.truncated is False
    assert turn.span.field == ""


def test_from_dict_refuses_to_invent_required_fields() -> None:
    with pytest.raises(TypeError):
        SourceSpan.from_dict({"source": "claude"})


# A dataclass ``session.py`` has never seen. If this round-trips, the
# serializer is genuinely generic rather than a per-class transcription.
@dataclass(slots=True)
class _LaterField(Wire):
    required: int
    tags: tuple[str, ...] = ()
    span: SourceSpan | None = None
    nested: list[Evidence] | None = None
    anything: object = None


def test_serializer_generalizes_to_a_dataclass_it_was_never_written_for() -> None:
    sample = _LaterField(
        required=7,
        tags=("a", "b"),
        span=SourceSpan("claude", "/p", 1, "f"),
        nested=[Evidence("q", "l")],
        anything={"free": ["form", 1, None]},
    )
    wire = sample.to_dict()

    assert wire == {
        "required": 7,
        "tags": ["a", "b"],
        "span": {"source": "claude", "path": "/p", "record_index": 1, "field": "f"},
        "nested": [{"quote": "q", "locator": "l"}],
        "anything": {"free": ["form", 1, None]},
    }
    assert _LaterField.from_dict(json.loads(json.dumps(wire))) == sample
    assert _LaterField.from_dict({"required": 1}) == _LaterField(1)


def test_schema_types_do_not_hand_write_serializers() -> None:
    """The anti-drift rule itself: no per-field to_dict/from_dict may return.

    ``SessionRecord.to_dict`` is the one sanctioned override -- it stamps
    ``schema_version`` onto the generic result and touches no field.
    """

    for cls in SCHEMA_TYPES:
        assert "from_dict" not in vars(cls), f"{cls.__name__} hand-writes from_dict"
        if cls is not SessionRecord:
            assert "to_dict" not in vars(cls), f"{cls.__name__} hand-writes to_dict"

    body = SessionRecord.to_dict.__code__
    assert not (set(body.co_names) & {f.name for f in fields(SessionRecord)})


# --------------------------------------------------------------------------
# published contract
# --------------------------------------------------------------------------

#: The frozen wire names. Adding a field is allowed and does not touch this
#: table; renaming or removing one must bump ``SCHEMA_VERSION`` and update
#: every consumer, and this test is where that gets noticed.
FROZEN_FIELDS: dict[str, tuple[str, ...]] = {
    "SourceSpan": ("source", "path", "record_index", "field"),
    "Turn": ("index", "role", "text", "at", "span", "cli", "truncated"),
    "ToolCall": (
        "ordinal",
        "name",
        "server",
        "arguments",
        "span",
        "call_id",
        "at",
        "ok",
        "result_text",
        "result_truncated",
    ),
    "FileMutation": (
        "ordinal",
        "path",
        "kind",
        "span",
        "unified_diff",
        "pre_image",
        "post_image",
        "at",
        "tool",
        "truncated",
    ),
    "DeliveredNode": (
        "node_id",
        "rank",
        "content",
        "scope",
        "level",
        "agent",
        "task",
        "created_at",
        "score",
        "bm25_score",
        "vector_score",
        "graph_score",
        "trigger_score",
        "methods",
        "delivery",
        "content_truncated",
    ),
    "RecallInteraction": (
        "ordinal",
        "query",
        "span",
        "recall_event_id",
        "scope",
        "depth",
        "max_results",
        "ambient_context",
        "delivered",
        "at",
        "turn_index",
        "record_index",
        "position",
        "truncated",
    ),
    "WriteInteraction": (
        "ordinal",
        "kind",
        "span",
        "node_id",
        "content",
        "context",
        "implicit_feedback",
        "prior_recall_ids",
        "supersedes",
        "transport_session_id",
        "at",
        "turn_index",
        "record_index",
        "position",
        "truncated",
    ),
    "SessionRecord": (
        "session_key",
        "source",
        "cli",
        "path",
        "cli_session_id",
        "model",
        "cwd",
        "repo",
        "git_branch",
        "git_commit",
        "started_at",
        "ended_at",
        "transport_session_ids",
        "linked_session_ids",
        "transcript_sha256",
        "record_count",
        "turns",
        "tool_calls",
        "file_mutations",
        "recalls",
        "writes",
        "parse_warnings",
    ),
    "Evidence": ("quote", "locator"),
    "Provenance": ("session_key", "transcript_sha256", "source_span"),
    "ProposedOp": ("kind", "payload", "provenance", "evidence"),
}


def test_published_field_names_are_frozen() -> None:
    assert SCHEMA_VERSION == 1, "renaming or dropping a field must bump this"
    assert {cls.__name__ for cls in SCHEMA_TYPES} == set(FROZEN_FIELDS), (
        "every published type must be pinned in FROZEN_FIELDS"
    )
    for cls in SCHEMA_TYPES:
        missing = set(FROZEN_FIELDS[cls.__name__]) - set(published_fields(cls))
        assert not missing, f"{cls.__name__} lost published field(s): {sorted(missing)}"


def test_added_fields_must_be_optional() -> None:
    """Additions stay backward compatible only if they carry a default."""

    for cls in SCHEMA_TYPES:
        pinned = FROZEN_FIELDS[cls.__name__]
        for f in fields(cls):  # type: ignore[arg-type]
            if f.name in pinned:
                continue
            has_default = f.default is not MISSING or f.default_factory is not MISSING
            assert has_default, f"new field {cls.__name__}.{f.name} needs a default"


# --------------------------------------------------------------------------
# tool-name normalization and derived views
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # claude: mcp__<server>__<tool>
        ("mcp__living-memory__memory_recall", ("living-memory", "memory_recall")),
        ("mcp__living-memory__memory_remember", ("living-memory", "memory_remember")),
        ("mcp__living-memory__memory_teach", ("living-memory", "memory_teach")),
        ("mcp__gitlab__list_issues", ("gitlab", "list_issues")),
        ("mcp__living-memory", ("living-memory", "")),
        # ae_chat / dotted hosts: <server>.<tool>
        ("living-memory.memory_recall", ("living-memory", "memory_recall")),
        ("living-memory.memory_teach", ("living-memory", "memory_teach")),
        (".memory_recall", (None, ".memory_recall")),
        ("living-memory.", (None, "living-memory.")),
        # plain built-ins and junk
        ("Bash", (None, "Bash")),
        ("edit_file", (None, "edit_file")),
        ("  Read  ", (None, "Read")),
        ("apply patch.py to x", (None, "apply patch.py to x")),
        ("", (None, "")),
        (None, (None, "")),
    ],
)
def test_normalize_tool_name(raw: str | None, expected: tuple[str | None, str]) -> None:
    assert normalize_tool_name(raw) == expected


def test_normalized_memory_calls_are_flagged() -> None:
    record = make_record()
    server, tool = normalize_tool_name("mcp__living-memory__memory_recall")
    assert (server, tool) == (record.tool_calls[0].server, record.tool_calls[0].name)
    assert record.tool_calls[0].is_memory_tool is True
    assert record.tool_calls[1].is_memory_tool is False


def test_derived_identifier_views() -> None:
    record = make_record()
    assert record.recall_event_ids == ("01M0BNNKQCYKSH4V4M8QRQ5MPY",)
    assert record.delivered_node_ids == (
        "01M0BKS48MVR2XHNKJRB2QF7M8",
        "01KV47B1HCYEY17D1E5RV6ZS3W",
    )
    assert record.recalls[0].delivered_node_ids == record.delivered_node_ids
    assert record.written_node_ids == (
        "01M0C000000000000000000000",
        "01M0C111111111111111111111",
    )
    assert record.mutated_paths == (
        "src/living_memory/postsession/session.py",
        "notes.md",
    )
    assert record.writes[0].feedback_applied is True
    assert record.writes[1].feedback_applied is False
    assert record.span(4, "payload").locator().endswith("#4:payload")


def test_summary_is_counts_only() -> None:
    record = make_record()
    blob = json.dumps(record.summary())
    assert "REDACTED" not in blob, "summary must never carry transcript content"
    assert record.summary()["recall_event_ids"] == 1
    assert record.summary()["delivered_nodes"] == 2
    assert record.summary()["written_nodes"] == 2
    assert record.summary()["file_mutations"] == 2


def test_package_exports_the_schema_without_its_siblings() -> None:
    """Importing the package must not require transcripts.py/corpus.py."""

    import living_memory.postsession as postsession

    assert postsession.SessionRecord is SessionRecord
    assert postsession.ProposedOp is ProposedOp
    assert postsession.SCHEMA_VERSION == SCHEMA_VERSION
    assert postsession.normalize_tool_name is normalize_tool_name
    for name in postsession.__all__:
        assert hasattr(postsession, name)
    with pytest.raises(AttributeError):
        postsession.definitely_not_exported


def test_package_forwards_to_a_submodule_once_it_lands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``transcripts``/``corpus`` arrive later and re-export through here."""

    import sys
    import types

    import living_memory.postsession as postsession

    stub = types.ModuleType("living_memory.postsession.transcripts")
    stub.load_transcript = lambda ref: ref  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "living_memory.postsession.transcripts", stub)

    assert postsession.load_transcript("ref") == "ref"
