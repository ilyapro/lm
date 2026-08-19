"""Adapter tests over redacted fixtures, one per transcript source.

The fixtures under ``tests/fixtures/postsession`` are redacted by
construction: every prose field is a ``REDACTED-*`` sentinel and every
identifier is synthetic. They mirror the *structure* measured on the real
corpus on 2026-08-19, including the awkward parts -- the compact
``memory_remember`` response that omits the content, Claude's spilled
``tool-results`` sidecar, codex's ``{"Ok"}``/``{"Err"}`` enum and its older
text-only MCP replies, gigacode's ``{"output": "<json>"}`` channel, and the AE
dashboard's ~250-character truncation of MCP payloads.
"""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from living_memory.postsession.session import (  # noqa: E402
    Evidence,
    ProposedOp,
    ProposedOpError,
    SourceSpan,
    normalize_tool_name,
)
from living_memory.postsession.transcripts import (  # noqa: E402
    ADAPTERS,
    Roots,
    discover_transcripts,
    load_transcript,
    render_structured_patch,
    repo_root,
)

FIXTURES = ROOT_DIR / "tests" / "fixtures" / "postsession"

CLAUDE_SID = "11111111-1111-4111-8111-111111111111"
CODEX_SID = "22222222-2222-4222-8222-222222222222"
GIGA_SID = "33333333-3333-4333-8333-333333333333"
DEEP_SID = "44444444-4444-4444-8444-444444444444"
AE_CODEX_SID = "55555555-5555-4555-8555-555555555555"
AGENT_STEM = "agent-a0000000000000001"
CHAT_ID = "c1700000000000-abcd"


def materialize_fixtures(tmp_path: Path) -> Roots:
    """Copy the fixture tree somewhere writable and resolve its paths.

    One fixture -- Claude's spilled ``tool-results`` stub -- must reference an
    absolute path, so it carries a ``__FIXTURE_ROOT__`` token that is rewritten
    here. Copying also keeps a test that mutates a transcript (the seal checks)
    from touching the checked-in fixtures.
    """

    target = tmp_path / "corpus"
    shutil.copytree(FIXTURES, target)
    placeholder = "__FIXTURE_ROOT__"
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if placeholder in text:
            path.write_text(text.replace(placeholder, str(target)), encoding="utf-8")
    return Roots.discover(home=target / "home", ae_root=target / "ae")


@pytest.fixture()
def roots(tmp_path: Path) -> Roots:
    return materialize_fixtures(tmp_path)


def refs_by_source(roots: Roots) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for ref in discover_transcripts(roots):
        grouped.setdefault(ref.source, []).append(ref)
    return grouped


def load_one(roots: Roots, source: str, needle: str = ""):
    for ref in ADAPTERS[source].discover(roots):
        if needle in str(ref.path):
            return load_transcript(ref)
    raise AssertionError(f"no {source} transcript matching {needle!r}")


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def test_every_fixture_file_is_committed() -> None:
    """A fixture that lives only in the working tree makes this suite a lie.

    Both halves of that trap have already sprung here. ``.gitignore``'s
    unanchored ``.claude/`` rule matches the fixture tree that mirrors
    ``~/.claude/projects``, and git never lists ignored paths, so ``git
    status`` stayed silent while the commit shipped without the Claude
    fixtures. The codex fixture then reached the gate untracked for the duller
    reason -- nobody staged it. Either way the working tree is green and a
    clean checkout is not, so assert what git actually carries rather than
    what happens to be on disk.

    Scoped to the two corpus roots the adapters read. ``cassettes/`` is a
    sibling's, and it is written by an opt-in live recording run, so an
    unstaged cassette is that node's failure to report, not this one's.
    """

    if not (ROOT_DIR / ".git").exists():
        # Already running from an export, which is the proof this test wants.
        pytest.skip("not a git checkout")
    tracked = set(
        subprocess.run(
            ["git", "-C", str(ROOT_DIR), "ls-files", "--", "tests/fixtures/postsession"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    )
    on_disk = {
        path.relative_to(ROOT_DIR).as_posix()
        for root in ("home", "ae")
        for path in (FIXTURES / root).rglob("*")
        if path.is_file()
    }
    assert on_disk - tracked == set()


def test_every_source_is_discovered(roots: Roots) -> None:
    grouped = refs_by_source(roots)
    assert sorted(grouped) == [
        "ae_chat",
        "ae_node_result",
        "claude",
        "codex",
        "deepseek",
        "gigacode",
    ]
    # Two codex roots must both be scanned: the user's own ~/.codex and the
    # AE-spawned codex_home. Scanning only the first drops most of the corpus.
    codex_ids = {ref.cli_session_id for ref in grouped["codex"]}
    assert codex_ids == {CODEX_SID, AE_CODEX_SID}
    # A Claude session and its subagent transcripts are separate sessions with
    # distinct keys that both name the parent session. (The remaining claude
    # fixtures are the split-coverage fillers, which have no subagents.)
    claude_keys = {ref.session_key for ref in grouped["claude"]}
    assert {f"claude:{CLAUDE_SID}", f"claude:{CLAUDE_SID}:{AGENT_STEM}"} <= claude_keys
    parent_refs = [ref for ref in grouped["claude"] if CLAUDE_SID in ref.session_key]
    assert len(parent_refs) == 2
    assert all(ref.linked_session_ids == (CLAUDE_SID,) for ref in parent_refs)
    # every claude transcript links to exactly the session it belongs to
    assert all(len(ref.linked_session_ids) == 1 for ref in grouped["claude"])


def test_discovery_is_deterministic(roots: Roots) -> None:
    first = [str(ref.path) for ref in discover_transcripts(roots)]
    second = [str(ref.path) for ref in discover_transcripts(roots)]
    assert first == second


def test_unknown_source_is_rejected(roots: Roots) -> None:
    with pytest.raises(KeyError):
        list(discover_transcripts(roots, ["nope"]))


# --------------------------------------------------------------------------
# claude
# --------------------------------------------------------------------------


def test_claude_recovers_recall_identifiers(roots: Roots) -> None:
    record = load_one(roots, "claude", f"{CLAUDE_SID}.jsonl")
    assert record.cli == "claude"
    assert record.cli_session_id == CLAUDE_SID
    assert record.model == "claude-opus-5"
    assert record.cwd == "/home/user/p/demo"
    assert record.repo == "/home/user/p/demo"
    assert record.git_branch == "demo-branch"

    assert [item.recall_event_id for item in record.recalls] == [
        "01AAAAAAAAAAAAAAAAAAAAAAA1",
        "01AAAAAAAAAAAAAAAAAAAAAAA2",
    ]
    first = record.recalls[0]
    assert first.query == "REDACTED-QUERY-A"
    assert first.scope == "project:demo"
    assert first.max_results == 5
    assert first.delivered_node_ids == (
        "01ND0EAAAAAAAAAAAAAAAAAAA1",
        "01ND0EAAAAAAAAAAAAAAAAAAA2",
    )
    top = first.delivered[0]
    assert (top.rank, top.score, top.bm25_score, top.vector_score) == (0, 0.91, 0.25, 0.5)
    assert top.methods == ("bm25", "vector")
    assert top.delivery == "full" and not top.content_truncated
    # a snippet delivery carries a content_ref, i.e. the agent saw less than the node
    assert first.delivered[1].content_truncated is True
    assert "mcpMeta.structuredContent" in first.span.locator()


def test_claude_follows_the_spilled_tool_result_sidecar(roots: Roots) -> None:
    record = load_one(roots, "claude", f"{CLAUDE_SID}.jsonl")
    spilled = record.recalls[1]
    assert spilled.recall_event_id == "01AAAAAAAAAAAAAAAAAAAAAAA2"
    assert spilled.delivered_node_ids == ("01ND0EAAAAAAAAAAAAAAAAAAA1",)
    assert spilled.span.field == "tool-results-sidecar"


def test_claude_recovers_writes_and_supersedes(roots: Roots) -> None:
    record = load_one(roots, "claude", f"{CLAUDE_SID}.jsonl")
    teach, remember = record.writes
    assert (teach.kind, teach.node_id) == ("teach", "01ND0EAAAAAAAAAAAAAAAAAAA9")
    assert teach.supersedes == "01ND0EAAAAAAAAAAAAAAAAAAA1"
    # the compact response omits the text, so it comes from the call arguments
    assert teach.content == "REDACTED-CORRECTION-TEXT"
    assert teach.feedback_applied is True
    assert teach.prior_recall_ids == ("01AAAAAAAAAAAAAAAAAAAAAAA1",)
    assert teach.transport_session_id == "f" * 32

    assert (remember.kind, remember.node_id) == ("remember", "01ND0EAAAAAAAAAAAAAAAAAAAB")
    assert remember.content == "REDACTED-REMEMBERED-FACT"
    # this record has no mcpMeta at all: the JSON-string toolUseResult is used
    assert remember.span.field == "toolUseResult"
    assert record.written_node_ids == (
        "01ND0EAAAAAAAAAAAAAAAAAAA9",
        "01ND0EAAAAAAAAAAAAAAAAAAAB",
    )
    assert set(record.transport_session_ids) == {"e" * 32, "f" * 32}


def test_claude_recovers_file_mutations(roots: Roots) -> None:
    record = load_one(roots, "claude", f"{CLAUDE_SID}.jsonl")
    assert record.mutated_paths == (
        "/home/user/p/demo/mod.py",
        "/home/user/p/demo/new.md",
    )
    edit, write = record.file_mutations
    assert edit.kind == "update" and edit.tool == "Edit"
    assert edit.unified_diff == (
        "--- a//home/user/p/demo/mod.py\n"
        "+++ b//home/user/p/demo/mod.py\n"
        "@@ -1,3 +1,3 @@\n HEADER\n-OLD_LINE\n+NEW_LINE\n FOOTER\n"
    )
    assert edit.pre_image == "HEADER\nOLD_LINE\nFOOTER\n"
    assert write.kind == "create" and write.post_image == "REDACTED-FILE-BODY\n"


def test_claude_positions_are_fractional_and_ordered(roots: Roots) -> None:
    record = load_one(roots, "claude", f"{CLAUDE_SID}.jsonl")
    positions = [item.position for item in record.recalls]
    assert all(0.0 <= value <= 1.0 for value in positions)
    assert positions == sorted(positions)
    # recall is the opening ritual, remember the closing one
    assert record.recalls[0].position < record.writes[-1].position
    assert [item.ordinal for item in record.recalls] == [0, 1]


def test_claude_keeps_turns_and_ignores_housekeeping_records(roots: Roots) -> None:
    record = load_one(roots, "claude", f"{CLAUDE_SID}.jsonl")
    texts = [turn.text for turn in record.turns]
    assert "REDACTED-USER-PROMPT" in texts
    assert "REDACTED-ASSISTANT-TEXT" in texts
    # ai-title / last-prompt / queue-operation records are not conversation
    assert not any("REDACTED-TITLE" in text for text in texts)
    assert not any("REDACTED-QUEUED-PROMPT" in text for text in texts)
    assert [turn.index for turn in record.turns] == list(range(len(record.turns)))


def test_claude_subagent_transcript_is_its_own_session(roots: Roots) -> None:
    record = load_one(roots, "claude", "subagents")
    assert record.session_key == f"claude:{CLAUDE_SID}:{AGENT_STEM}"
    assert record.linked_session_ids == (CLAUDE_SID,)
    assert record.recalls == [] and record.writes == []
    assert any("REDACTED-SUBAGENT-ANSWER" in turn.text for turn in record.turns)


# --------------------------------------------------------------------------
# codex
# --------------------------------------------------------------------------


def test_codex_recovers_session_metadata(roots: Roots) -> None:
    record = load_one(roots, "codex", CODEX_SID)
    assert record.cli == "codex"
    assert record.cli_session_id == CODEX_SID
    assert record.model == "gpt-5.6-sol"
    assert record.cwd == "/home/user/p/demo"
    assert record.git_branch == "codex-branch"
    assert record.git_commit == "0123456789abcdef"
    assert record.started_at == "2026-08-01T11:00:00.000Z"
    assert record.ended_at == "2026-08-01T11:03:01.000Z"


def test_codex_recovers_recalls_including_ambient_context(roots: Roots) -> None:
    record = load_one(roots, "codex", CODEX_SID)
    assert [item.recall_event_id for item in record.recalls] == [
        "01BBBBBBBBBBBBBBBBBBBBBBB1",
        "01BBBBBBBBBBBBBBBBBBBBBBB2",
    ]
    first = record.recalls[0]
    # ambient_context is codex-only: the Claude host never sends it
    assert first.ambient_context["session_id"] == "demo-node-run"
    assert first.ambient_context["agent"] == "codex"
    assert first.depth == "causal"
    assert first.delivered_node_ids == (
        "01ND0EBBBBBBBBBBBBBBBBBBB1",
        "01ND0EBBBBBBBBBBBBBBBBBBB2",
    )
    assert first.delivered[0].trigger_score == 0.1
    # the older rollout shape has no structuredContent, only the text block
    assert record.recalls[1].delivered_node_ids == ("01ND0EBBBBBBBBBBBBBBBBBBB1",)


def test_codex_err_arm_yields_no_interaction(roots: Roots) -> None:
    record = load_one(roots, "codex", CODEX_SID)
    failed = [call for call in record.tool_calls if call.call_id == "call_r3"]
    assert len(failed) == 1 and failed[0].ok is False
    assert "01BBBBBBBBBBBBBBBBBBBBBBB3" not in record.recall_event_ids
    assert len(record.recalls) == 2


def test_codex_recovers_writes_and_patches(roots: Roots) -> None:
    record = load_one(roots, "codex", CODEX_SID)
    (write,) = record.writes
    assert write.node_id == "01ND0EBBBBBBBBBBBBBBBBBBB9"
    assert write.content == "REDACTED-CODEX-REMEMBERED-FACT"
    assert write.prior_recall_ids == (
        "01BBBBBBBBBBBBBBBBBBBBBBB1",
        "01BBBBBBBBBBBBBBBBBBBBBBB2",
    )
    assert write.feedback_applied is True
    assert record.transport_session_ids == ("a" * 32,)

    added, updated = sorted(record.file_mutations, key=lambda item: item.path)
    assert (added.path, added.kind) == ("/home/user/p/demo/added.py", "create")
    assert added.post_image == "REDACTED-ADDED-FILE-BODY\n"
    assert updated.kind == "update"
    assert updated.unified_diff.startswith("@@ -1,3 +1,3 @@")
    assert updated.tool == "apply_patch"


def test_codex_keeps_user_and_agent_turns(roots: Roots) -> None:
    record = load_one(roots, "codex", CODEX_SID)
    assert [(turn.role, turn.text) for turn in record.turns] == [
        ("user", "REDACTED-CODEX-USER-MESSAGE"),
        ("assistant", "REDACTED-CODEX-AGENT-MESSAGE"),
    ]


def test_codex_shell_call_is_paired_with_its_output(roots: Roots) -> None:
    record = load_one(roots, "codex", CODEX_SID)
    (patch_call,) = [c for c in record.tool_calls if c.name == "apply_patch"]
    assert patch_call.ok is True
    assert patch_call.result_text.startswith("Exit code: 0")


# --------------------------------------------------------------------------
# ae_chat and ae_node_result
# --------------------------------------------------------------------------


def test_ae_chat_salvages_recall_ids_from_truncated_payloads(roots: Roots) -> None:
    record = load_one(roots, "ae_chat", CHAT_ID)
    assert record.cli == "ae"
    assert record.session_key == f"ae_chat:{CHAT_ID}"
    assert [item.recall_event_id for item in record.recalls] == [
        "01DDDDDDDDDDDDDDDDDDDDDDD1",
        None,
    ]
    # every dashboard MCP payload is truncated, and the record says so
    assert all(item.truncated for item in record.recalls)
    assert record.recalls[0].query == "REDACTED-AE-QUERY"
    (write,) = record.writes
    assert write.node_id == "01ND0EDDDDDDDDDDDDDDDDDDD9"
    assert write.truncated is True


def test_ae_chat_links_to_the_cli_session_that_served_it(roots: Roots) -> None:
    record = load_one(roots, "ae_chat", CHAT_ID)
    # meta.json's sessions_by_provider is the fidelity path: the claude
    # transcript holds the untruncated payloads this chat only summarised
    assert CLAUDE_SID in record.linked_session_ids
    assert record.model == "claude-opus-5"
    assert record.started_at == "2026-08-05T12:00:00Z"


def test_ae_node_result_recovers_its_join_key(roots: Roots) -> None:
    record = load_one(roots, "ae_node_result", "demo-goal")
    assert record.session_key == "ae_node:demo:demo-goal"
    assert record.cli == "codex"
    assert record.cli_session_id == AE_CODEX_SID
    assert record.linked_session_ids == (AE_CODEX_SID,)
    assert record.model == "gpt-5.6-sol"
    assert record.cwd == "/home/user/p/demo/.worktrees/_node_exec_demo"
    # a worktree collapses to the repository it belongs to
    assert record.repo == "/home/user/p/demo"
    assert any("REDACTED-NODE-SUMMARY" in turn.text for turn in record.turns)
    assert record.parse_warnings


# --------------------------------------------------------------------------
# gigacode and deepseek
# --------------------------------------------------------------------------


def test_gigacode_reads_the_output_string_channel(roots: Roots) -> None:
    record = load_one(roots, "gigacode", GIGA_SID)
    assert record.cli == "gigacode"
    assert record.model == "CodeChat"
    (recall,) = record.recalls
    assert recall.recall_event_id == "01CCCCCCCCCCCCCCCCCCCCCCC1"
    assert recall.delivered_node_ids == ("01ND0ECCCCCCCCCCCCCCCCCCC1",)
    assert recall.query == "REDACTED-GIGA-QUERY"
    assert [turn.role for turn in record.turns] == ["user", "assistant"]


def test_deepseek_streams_one_json_document(roots: Roots) -> None:
    record = load_one(roots, "deepseek", DEEP_SID)
    assert record.cli == "deepseek"
    assert record.cli_session_id == DEEP_SID
    assert record.model == "deepseek-v4-pro"
    assert record.cwd == "/home/user/p/demo"
    assert record.started_at == "2026-08-04T13:00:00.000000000Z"
    assert record.record_count == 8
    (mutation,) = record.file_mutations
    assert mutation.path == "/home/user/p/demo/mod.py"
    assert (mutation.pre_image, mutation.post_image) == ("OLD_LINE", "NEW_LINE")
    assert [turn.text for turn in record.turns] == [
        "REDACTED-DEEPSEEK-PROMPT",
        "REDACTED-DEEPSEEK-ANSWER",
    ]


def test_deepseek_recovers_memory_identifiers(roots: Roots) -> None:
    """Deepseek stores tool blocks in Anthropic's shape, so an MCP reply lands
    in the same form the Claude host records -- and must be recovered, not left
    sitting in ``result_text``.

    The two round trips exercise both shapes this host emits: the recall result
    is a list of ``text`` blocks, the remember result a bare JSON string.
    """

    record = load_one(roots, "deepseek", DEEP_SID)
    (recall,) = record.recalls
    assert recall.recall_event_id == "01EEEEEEEEEEEEEEEEEEEEEEE1"
    assert recall.query == "REDACTED-DEEPSEEK-QUERY"
    assert recall.scope == "project:demo"
    assert recall.delivered_node_ids == ("01ND0EEEEEEEEEEEEEEEEEEEE1",)
    assert recall.delivered[0].methods == ("bm25", "vector")

    (write,) = record.writes
    assert (write.kind, write.node_id) == ("remember", "01ND0EEEEEEEEEEEEEEEEEEEE9")
    # the compact reply omits the text, so it is recovered from the arguments
    assert write.content == "REDACTED-DEEPSEEK-REMEMBERED-FACT"
    assert write.prior_recall_ids == ("01EEEEEEEEEEEEEEEEEEEEEEE1",)
    assert write.feedback_applied is True
    assert record.transport_session_ids == ("d" * 32,)
    # recall opens the session, remember closes it
    assert recall.position < write.position


def test_deepseek_scanner_survives_chunk_boundaries(tmp_path: Path) -> None:
    """The incremental decoder must not depend on where a read lands."""

    from living_memory.postsession.transcripts import _JsonMemberScanner

    document = {
        "schema_version": "1",
        "metadata": {"id": "x", "model": "m"},
        "messages": [{"role": "user", "content": [{"type": "text", "text": "t" * 40}]}] * 25,
        "artifacts": [],
    }
    path = tmp_path / "one.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    for chunk_size in (4, 7, 64, 4096):
        with _JsonMemberScanner(path, chunk_size=chunk_size) as scanner:
            assert scanner.value("metadata") == {"id": "x", "model": "m"}
        with _JsonMemberScanner(path, chunk_size=chunk_size) as scanner:
            assert len(list(scanner.array("messages"))) == 25


# --------------------------------------------------------------------------
# streaming invariant
# --------------------------------------------------------------------------


class _GuardedFile:
    """A file proxy that records the size of every whole-buffer read.

    Both slurp routes are proxied. ``read()`` with no size is the obvious one;
    ``readlines()`` is the one that hides, because it returns a list of lines
    and so *looks* line-oriented while having already materialised the entire
    file. Both are recorded as ``None`` -- an unbounded read -- and the test
    below rejects any ``None``.
    """

    def __init__(self, handle, sizes: list[int | None]):
        self._handle = handle
        self._sizes = sizes

    def read(self, size: int = -1):
        self._sizes.append(None if size is None or size < 0 else size)
        return self._handle.read(size)

    def readlines(self, hint: int = -1):
        self._sizes.append(None if hint is None or hint < 0 else hint)
        return self._handle.readlines(hint)

    def __iter__(self):
        return iter(self._handle)

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._handle, name)


def test_adapters_never_slurp_a_transcript(tmp_path: Path, monkeypatch) -> None:
    """A big transcript must be consumed in bounded reads, not read whole.

    Real transcripts reach 7 MB (Claude) and 102 MB (``result.md``); an adapter
    that called ``handle.read()`` would make the extraction stage unrunnable on
    the accumulated corpus.
    """

    session = "99999999-9999-4999-8999-999999999999"
    path = (
        tmp_path / "home" / ".claude" / "projects" / "-home-user-p-demo" / f"{session}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    filler = "y" * 4000
    with open(path, "w", encoding="utf-8") as handle:
        for index in range(3000):
            handle.write(
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": f"n{index}",
                        "sessionId": session,
                        "cwd": "/home/user/p/demo",
                        "timestamp": "2026-08-01T10:00:00.000Z",
                        "message": {
                            "role": "assistant",
                            "model": "claude-opus-5",
                            "content": [{"type": "text", "text": filler}],
                        },
                    }
                )
                + "\n"
            )
    assert path.stat().st_size > 12_000_000

    sizes: list[int | None] = []
    real_open = builtins.open

    def guarded(file, *args, **kwargs):
        handle = real_open(file, *args, **kwargs)
        if str(file) == str(path):
            return _GuardedFile(handle, sizes)
        return handle

    monkeypatch.setattr(builtins, "open", guarded)
    roots = Roots.discover(home=tmp_path / "home", ae_root=tmp_path / "missing-ae")
    (ref,) = ADAPTERS["claude"].discover(roots)
    record = load_transcript(ref)
    monkeypatch.undo()

    assert record.record_count == 3000
    assert len(record.turns) == 3000
    assert all(size is not None and size <= (1 << 20) for size in sizes)


def test_deepseek_single_document_is_read_in_bounded_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    """The one source that cannot stream by line must still not slurp.

    A deepseek session is a single JSON document, so ``json.load`` would be the
    obvious implementation and would hold the whole file plus its decoded form
    in memory at once. The incremental scanner must instead read in bounded
    chunks -- and unlike the line-oriented adapters, this one really does call
    ``read(size)``, so the recorded sizes are non-empty and the bound below is
    actually exercised rather than vacuously true.
    """

    session = "77777777-7777-4777-8777-777777777777"
    path = tmp_path / "home" / ".deepseek" / "sessions" / f"{session}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "metadata": {"id": session, "model": "deepseek-v4-pro"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "z" * 4000}]}
            for _ in range(2000)
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    assert path.stat().st_size > 8_000_000

    sizes: list[int | None] = []
    real_open = builtins.open

    def guarded(file, *args, **kwargs):
        handle = real_open(file, *args, **kwargs)
        if str(file) == str(path):
            return _GuardedFile(handle, sizes)
        return handle

    monkeypatch.setattr(builtins, "open", guarded)
    roots = Roots.discover(home=tmp_path / "home", ae_root=tmp_path / "missing-ae")
    (ref,) = ADAPTERS["deepseek"].discover(roots)
    record = load_transcript(ref)
    monkeypatch.undo()

    assert record.record_count == 2000
    assert record.model == "deepseek-v4-pro"
    assert sizes, "the scanner never read anything"
    assert all(size is not None and size <= (1 << 20) for size in sizes)


def test_pending_call_state_stays_bounded(tmp_path: Path) -> None:
    """Unanswered tool calls must not accumulate for the length of a session.

    A crashed or interrupted session leaves tool calls with no result. The
    adapter caps how many it keeps waiting for; the cap must not break a call
    that *is* answered right after the flood.
    """

    from living_memory.postsession.transcripts import PENDING_CALL_LIMIT

    session = "88888888-8888-4888-8888-888888888888"
    path = (
        tmp_path / "home" / ".claude" / "projects" / "-home-user-p-demo" / f"{session}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(PENDING_CALL_LIMIT * 2):
        records.append(
            {
                "type": "assistant",
                "uuid": f"orphan{index}",
                "sessionId": session,
                "cwd": "/home/user/p/demo",
                "timestamp": "2026-08-01T10:00:00.000Z",
                "message": {"role": "assistant", "model": "claude-opus-5", "content": [
                    {"type": "tool_use", "id": f"toolu_orphan{index}", "name": "Edit",
                     "input": {"file_path": f"/home/user/p/demo/f{index}.py"}}]},
            }
        )
    records.append({
        "type": "assistant", "uuid": "answered", "sessionId": session,
        "timestamp": "2026-08-01T10:01:00.000Z",
        "message": {"role": "assistant", "model": "claude-opus-5", "content": [
            {"type": "tool_use", "id": "toolu_answered", "name": "Write",
             "input": {"file_path": "/home/user/p/demo/late.md"}}]}})
    records.append({
        "type": "user", "uuid": "result", "sessionId": session,
        "timestamp": "2026-08-01T10:01:01.000Z",
        "toolUseResult": {"type": "create", "filePath": "/home/user/p/demo/late.md",
                          "content": "REDACTED-LATE-BODY\n", "structuredPatch": []},
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_answered", "content": "ok"}]}})
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    roots = Roots.discover(home=tmp_path / "home", ae_root=tmp_path / "missing-ae")
    (ref,) = ADAPTERS["claude"].discover(roots)
    record = load_transcript(ref)
    assert len(record.tool_calls) == PENDING_CALL_LIMIT * 2 + 1
    # the orphans produced no mutation; the one that was answered did
    assert record.mutated_paths == ("/home/user/p/demo/late.md",)
    assert record.file_mutations[0].post_image == "REDACTED-LATE-BODY\n"


def test_result_md_body_is_bounded(tmp_path: Path) -> None:
    """``result.md`` reaches 102 MB, so only a bounded tail is retained."""

    from living_memory.postsession.transcripts import RESULT_TAIL_LINES

    goal = tmp_path / "ae" / "projects" / "demo" / "state" / "goals" / "big"
    goal.mkdir(parents=True)
    body = "".join(f"line-{index}\n" for index in range(RESULT_TAIL_LINES * 5))
    (goal / "result.md").write_text("OpenAI Codex v1\nsession id: abc\n" + body, encoding="utf-8")
    roots = Roots.discover(home=tmp_path / "missing-home", ae_root=tmp_path / "ae")
    (ref,) = ADAPTERS["ae_node_result"].discover(roots)
    record = load_transcript(ref)
    (turn,) = record.turns
    assert turn.truncated is True
    assert turn.text.count("\n") + 1 == RESULT_TAIL_LINES
    assert turn.text.endswith(f"line-{RESULT_TAIL_LINES * 5 - 1}")
    assert record.record_count == RESULT_TAIL_LINES * 5 + 2


# --------------------------------------------------------------------------
# shared schema
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("mcp__living-memory__memory_recall", ("living-memory", "memory_recall")),
        ("living-memory.memory_recall", ("living-memory", "memory_recall")),
        ("Bash", (None, "Bash")),
        ("", (None, "")),
    ],
)
def test_tool_name_normalization(raw: str, expected: tuple[str | None, str]) -> None:
    assert normalize_tool_name(raw) == expected


def test_source_span_round_trips() -> None:
    span = SourceSpan("claude", "/tmp/a.jsonl", 42, "mcpMeta.structuredContent")
    assert span.locator() == "claude:/tmp/a.jsonl#42:mcpMeta.structuredContent"
    assert SourceSpan.parse(span.locator()) == span
    assert SourceSpan.from_dict(span.to_dict()) == span


@pytest.mark.parametrize("cwd", ["/home/user/p/demo/.worktrees/x", "/home/user/p/demo/"])
def test_repo_root_collapses_worktrees(cwd: str) -> None:
    assert repo_root(cwd) == "/home/user/p/demo"


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    from living_memory.postsession.transcripts import sha256_file

    path = tmp_path / "blob.bin"
    path.write_bytes(b"x" * 3_000_000)
    assert sha256_file(path, chunk_size=997) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_render_structured_patch_returns_none_without_hunks() -> None:
    assert render_structured_patch("a.py", []) is None
    assert render_structured_patch("a.py", None) is None


def test_proposed_op_requires_provenance_and_evidence(roots: Roots) -> None:
    record = load_one(roots, "claude", f"{CLAUDE_SID}.jsonl")
    record.transcript_sha256 = "deadbeef"
    op = ProposedOp.for_session(
        "remember",
        {"content": "REDACTED-FACT", "context": {"scope": "project:demo"}},
        record,
        record.recalls[0].span,
        [Evidence("REDACTED-QUOTE", record.recalls[0].span.locator())],
    ).validate()
    assert op.provenance.session_key == f"claude:{CLAUDE_SID}"
    assert op.provenance.transcript_sha256 == "deadbeef"
    assert ProposedOp.from_dict(op.to_dict()) == op

    for broken in (
        ProposedOp("bogus", {"a": 1}, op.provenance, op.evidence),  # type: ignore[arg-type]
        ProposedOp("remember", {}, op.provenance, op.evidence),
        ProposedOp("remember", {"a": 1}, op.provenance, ()),
        ProposedOp("remember", {"a": 1}, op.provenance, (Evidence(" ", "x"),)),
        ProposedOp("remember", {"a": 1}, op.provenance, (Evidence("q", ""),)),
    ):
        with pytest.raises(ProposedOpError):
            broken.validate()


def test_session_summary_is_counts_only(roots: Roots) -> None:
    record = load_one(roots, "claude", f"{CLAUDE_SID}.jsonl")
    summary = record.summary()
    blob = json.dumps(summary)
    assert "REDACTED" not in blob
    assert summary["recall_event_ids"] == 2
    assert summary["written_nodes"] == 2
