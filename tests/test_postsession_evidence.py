"""What the offline evidence assembler must never get wrong.

Each test below pins one property that, if it broke, would break the honesty
or the idempotency of retroactive grounded credit:

* **determinism** -- the server's ledger key is
  ``(recall_event_id, evidence_sha256)``. An assembler that reorders or
  reformats items on a rerun re-credits the same session under a fresh key;
* **prose exclusion** -- an assistant turn can quote a recalled node verbatim.
  If prose counted as evidence, every recall would ground on the agent's own
  restatement of it, which is the self-marking circularity this path exists to
  escape;
* **memory-tool exclusion** -- a ``memory_recall`` result *is* the node text,
  so a single leaked result payload grounds every delivered node on itself;
* **redaction** -- a bounded evidence document leaves the machine, so the
  redactor must run before the caps are measured;
* **bounds** -- what the caps drop has to be reported, because a silent
  truncation reads as "we submitted everything";
* **a real session** -- fixtures prove the rules, a real train-split transcript
  proves the rules fire on transcripts nobody wrote for this test.

The last one reads the *sealed* corpus and only ``train`` rows. It never opens
a ``holdout`` session: that split is reserved for the final field measurement,
and a test that peeks at it spends the only unbiased sample the project has.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os

import pytest

from living_memory import attestation
from living_memory.attestation import (
    EVIDENCE_MAX_ITEM_CHARS,
    EVIDENCE_MAX_ITEMS,
    EVIDENCE_MAX_TOTAL_CHARS,
    canonical_evidence,
)
from living_memory.postsession import corpus as corpus_mod
from living_memory.postsession import evidence as evidence_mod
from living_memory.postsession.evidence import (
    CATEGORIES,
    COMMAND,
    DIFF_HUNK,
    FILE_WRITE,
    assemble_evidence,
    command_text,
    proposed_attestations,
    split_diff_hunks,
)
from living_memory.postsession.session import (
    DeliveredNode,
    FileMutation,
    RecallInteraction,
    SessionRecord,
    SourceSpan,
    ToolCall,
    Turn,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
TRACKED_MANIFEST = ROOT_DIR / "artifacts" / "post-session" / "corpus.json"

#: Where a locally built session index may be found, in lookup order: an
#: explicit override, the gitignored path beside the manifest, then a machine
#: cache outside the repository. The index holds thousands of absolute home
#: paths, so it is never committed; a machine that has not run
#: ``scripts/postsession_corpus.py --build`` skips the real-session tests
#: rather than inventing a substitute for real transcripts.
INDEX_ENV = "LM_POSTSESSION_INDEX"
CACHED_INDEX = (
    Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    / "living-memory"
    / "post-session"
    / "corpus-index.jsonl"
)

# --------------------------------------------------------------------------
# fixtures -- synthetic content, planted sentinels
# --------------------------------------------------------------------------

SESSION_KEY = "claude:11111111-1111-4111-8111-111111111111"
TRANSCRIPT_PATH = "/home/sfx/.claude/projects/-home-sfx-p-lm/s.jsonl"
TRANSCRIPT_SHA = "b" * 64
RECALL_EVENT_ID = "01JZZZZZZZZZZZZZZZZZZZZZZZ"

#: Distinctive tokens that appear ONLY in prose (the recalled node, the
#: assistant turn quoting it, the user prompt). None may reach an item.
NODE_SENTINEL = "zarquonratchet"
ASSISTANT_SENTINEL = "quillfeatherprose"
USER_SENTINEL = "brambleknotprompt"
#: Distinctive token that appears ONLY inside a memory-tool result payload.
MEMORY_RESULT_SENTINEL = "thimblewickpayload"
#: Distinctive tokens that appear ONLY inside artifacts, and so must survive.
DIFF_SENTINEL = "ledger_key_is_unique"
COMMAND_SENTINEL = "test_usage_attestation"

#: A deliberately FAKE credential planted in a command's output to prove the
#: redactor runs before the caps are measured. It is assembled from marker
#: words at import time and has never been a real key anywhere; the repo's
#: merge gate reads *unmarked* synthetic credentials as a leak.
FAKE_TOKEN = "sk-" + "EXAMPLE" + "NOTAREAL" + "SECRET" + "000000000000"
FAKE_ENV_ASSIGNMENT = "LM_AUTH_TOKEN=" + "FAKE" + "NOTAREALVALUE"

DIFF_TWO_HUNKS = f"""--- a/src/living_memory/storage.py
+++ b/src/living_memory/storage.py
@@ -1470,7 +1470,7 @@ class MemoryStore:
     def mark_recall_event_feedback(self, event_id: str, trace_id: str) -> None:
-        # legacy overwrite branch
+        # {DIFF_SENTINEL}: the flip credits linked_count exactly once
         with self._connect() as conn:
             conn.execute("UPDATE recall_events SET feedback_applied = 1")
@@ -1500,6 +1502,8 @@ class MemoryStore:
+    def find_recall_attestation(self, event_id: str, digest: str) -> None:
+        return None
"""

WHOLE_FILE_BODY = "\n".join(
    [
        "def claim_recall_attestation(event_id, digest):",
        '    """Take the ledger key or return None."""',
        "    return _insert_or_none(event_id, digest)",
    ]
)

PRE_IMAGE = "alpha = 1\nbeta = 2\ngamma = 3\n"
POST_IMAGE = "alpha = 1\nbeta = 22222\ngamma = 3\n"


def make_record() -> SessionRecord:
    """A session with prose, memory calls, artifacts and a planted secret."""

    def span(index: int, field: str) -> SourceSpan:
        return SourceSpan("claude", TRANSCRIPT_PATH, index, field)

    node = DeliveredNode(
        node_id="01KRV7M4YS7FD6YMS96QK19VCN",
        rank=0,
        content=(
            f"The {NODE_SENTINEL} invariant: an attestation is idempotent per "
            "(recall_event_id, evidence_sha256), never per recall_event_id."
        ),
        scope="project:lm",
    )
    return SessionRecord(
        session_key=SESSION_KEY,
        source="claude",
        cli="claude",
        path=TRANSCRIPT_PATH,
        transcript_sha256=TRANSCRIPT_SHA,
        record_count=12,
        turns=[
            Turn(0, "user", f"Fix the ledger, see {USER_SENTINEL}.", None, span(0, "")),
            Turn(
                1,
                "assistant",
                # The agent restates the recalled node verbatim. This is
                # exactly the text that must NOT become evidence.
                f"Recalled: the {NODE_SENTINEL} invariant says an attestation "
                f"is idempotent per (recall_event_id, evidence_sha256). "
                f"{ASSISTANT_SENTINEL}: applying it now.",
                None,
                span(2, ""),
            ),
        ],
        tool_calls=[
            ToolCall(
                ordinal=0,
                name="memory_recall",
                server="living-memory",
                arguments={"query": f"ledger idempotency {USER_SENTINEL}"},
                span=span(1, "message.content.tool_use"),
                result_text=(
                    '{"results": [{"content": "The '
                    + NODE_SENTINEL
                    + " invariant ... "
                    + MEMORY_RESULT_SENTINEL
                    + '"}]}'
                ),
            ),
            ToolCall(
                ordinal=1,
                name="Read",
                server=None,
                arguments={"file_path": "/home/sfx/p/lm/src/living_memory/storage.py"},
                span=span(3, "message.content.tool_use"),
                result_text="1  from __future__ import annotations",
            ),
            ToolCall(
                ordinal=2,
                name="Bash",
                server=None,
                arguments={
                    "command": f"python3 -m pytest -q tests/{COMMAND_SENTINEL}.py",
                    "description": "run the attestation tests",
                },
                span=span(4, "message.content.tool_use"),
                result_text=(
                    "12 passed in 3.41s\n"
                    f"env: {FAKE_ENV_ASSIGNMENT}\n"
                    f"token echoed by the harness: {FAKE_TOKEN}\n"
                    "wrote /home/sfx/p/lm/artifacts/attest.json"
                ),
            ),
            ToolCall(
                ordinal=3,
                name="memory_remember",
                server="living-memory",
                arguments={"content": f"{NODE_SENTINEL} closure note"},
                span=span(9, "message.content.tool_use"),
                result_text='{"node_id": "01KRV7M4YS7FD6YMS96QK19VCP"}',
            ),
        ],
        file_mutations=[
            FileMutation(
                ordinal=0,
                path="src/living_memory/storage.py",
                kind="update",
                span=span(5, "toolUseResult"),
                unified_diff=DIFF_TWO_HUNKS,
                tool="Edit",
            ),
            FileMutation(
                ordinal=1,
                path="src/living_memory/attestation.py",
                kind="create",
                span=span(6, "toolUseResult"),
                post_image=WHOLE_FILE_BODY,
                tool="Write",
            ),
            FileMutation(
                ordinal=2,
                path="src/living_memory/models.py",
                kind="update",
                span=span(7, "toolUseResult"),
                pre_image=PRE_IMAGE,
                post_image=POST_IMAGE,
                tool="Write",
            ),
            FileMutation(
                ordinal=3,
                path="docs/architecture.md",
                kind="unknown",
                span=span(8, "toolUseResult"),
                tool="Edit",
            ),
        ],
        recalls=[
            RecallInteraction(
                ordinal=0,
                query=f"ledger idempotency {USER_SENTINEL}",
                span=span(1, "mcpMeta.structuredContent"),
                recall_event_id=RECALL_EVENT_ID,
                scope="project:lm",
                delivered=[node],
            ),
            RecallInteraction(
                ordinal=1,
                query="anchor reinforcement",
                span=span(10, "mcpMeta.structuredContent"),
                recall_event_id=None,  # the host never captured the event id
            ),
        ],
    )


def big_record(mutations: int = 40, hunks: int = 12, commands: int = 40) -> SessionRecord:
    """A session far larger than the caps, built from cap-sized artifacts."""

    filler = "\n".join(f"+    line_{n} = compute_{n}(payload)" for n in range(40))
    diff = "\n".join(
        [f"--- a/src/mod_{{index}}.py", f"+++ b/src/mod_{{index}}.py"]
        + [f"@@ -{10 * h},0 +{10 * h},40 @@\n{filler}" for h in range(hunks)]
    )
    return SessionRecord(
        session_key=SESSION_KEY,
        source="claude",
        cli="claude",
        path=TRANSCRIPT_PATH,
        transcript_sha256=TRANSCRIPT_SHA,
        file_mutations=[
            FileMutation(
                ordinal=index,
                path=f"src/mod_{index}.py",
                kind="update",
                span=SourceSpan("claude", TRANSCRIPT_PATH, index, "toolUseResult"),
                unified_diff=diff.format(index=index),
            )
            for index in range(mutations)
        ],
        tool_calls=[
            ToolCall(
                ordinal=index,
                name="Bash",
                server=None,
                arguments={"command": f"python3 -m pytest -q tests/test_{index}.py"},
                span=SourceSpan("claude", TRANSCRIPT_PATH, 100 + index, "tool_use"),
                result_text="\n".join(
                    f"tests/test_{index}.py::case_{n} PASSED" for n in range(60)
                ),
            )
            for index in range(commands)
        ],
    )


@pytest.fixture()
def record() -> SessionRecord:
    return make_record()


# --------------------------------------------------------------------------
# the caps are the server's
# --------------------------------------------------------------------------


def test_the_caps_are_the_servers_own_objects() -> None:
    """Re-declaring a cap here is how client and server silently diverge."""

    assert evidence_mod.EVIDENCE_MAX_ITEMS is attestation.EVIDENCE_MAX_ITEMS
    assert evidence_mod.EVIDENCE_MAX_ITEM_CHARS is attestation.EVIDENCE_MAX_ITEM_CHARS
    assert evidence_mod.EVIDENCE_MAX_TOTAL_CHARS is attestation.EVIDENCE_MAX_TOTAL_CHARS


def test_the_module_is_registered_for_lazy_import() -> None:
    """``postsession.assemble_evidence`` must resolve like its siblings do."""

    from living_memory import postsession

    assert "evidence" in postsession._LAZY_SUBMODULES
    assert postsession.assemble_evidence is assemble_evidence
    assert "evidence" in dir(postsession)


# --------------------------------------------------------------------------
# (a) determinism
# --------------------------------------------------------------------------


def test_assembly_is_byte_identical_across_runs(record: SessionRecord) -> None:
    """Same SessionRecord -> same items -> same ledger key, forever."""

    first = assemble_evidence(record)
    second = assemble_evidence(make_record())

    assert first.texts == second.texts
    assert first.spans == second.spans
    assert first.evidence_sha256 == second.evidence_sha256
    assert first.summary() == second.summary()
    assert first.texts, "the fixture session must produce evidence at all"


def test_the_digest_is_the_one_the_server_will_compute(record: SessionRecord) -> None:
    """Every item is already canonical, so no item changes shape server-side."""

    bundle = assemble_evidence(record)
    canonical, digest = canonical_evidence(bundle.texts)

    assert canonical == bundle.texts
    assert digest == bundle.evidence_sha256


def test_ordering_puts_breadth_before_depth(record: SessionRecord) -> None:
    """The first items must span origins, not exhaust one file's hunks."""

    bundle = assemble_evidence(big_record(mutations=5, hunks=6, commands=5))
    first_mutations = [item for item in bundle.items if item.category == DIFF_HUNK][:5]

    assert len({item.origin for item in first_mutations}) == 5
    assert {item.category for item in bundle.items[:4]} == {DIFF_HUNK, COMMAND}


# --------------------------------------------------------------------------
# (b) prose is never evidence
# --------------------------------------------------------------------------


def test_prose_never_becomes_evidence(record: SessionRecord) -> None:
    """The assistant quotes the recalled node verbatim; the diffs do not."""

    bundle = assemble_evidence(record)
    joined = "\n".join(bundle.texts)

    assert NODE_SENTINEL in record.turns[1].text  # the fixture really quotes it
    assert NODE_SENTINEL not in joined
    assert ASSISTANT_SENTINEL not in joined
    assert USER_SENTINEL not in joined
    assert bundle.excluded["turns"] == len(record.turns) == 2
    # ...while the artifacts the session produced are all still there.
    assert DIFF_SENTINEL in joined
    assert COMMAND_SENTINEL in joined


# --------------------------------------------------------------------------
# (c) memory-tool payloads are never evidence
# --------------------------------------------------------------------------


def test_memory_tool_payloads_never_become_evidence(record: SessionRecord) -> None:
    """A recall result carries the node text: grounding it grounds itself."""

    bundle = assemble_evidence(record)
    joined = "\n".join(bundle.texts)

    assert MEMORY_RESULT_SENTINEL in (record.tool_calls[0].result_text or "")
    assert MEMORY_RESULT_SENTINEL not in joined
    assert "memory_recall" not in joined
    assert "memory_remember" not in joined
    assert bundle.excluded["memory_tool_calls"] == 2


def test_tool_calls_without_a_command_are_counted_not_guessed(
    record: SessionRecord,
) -> None:
    """``Read`` has no command; rendering its arguments would re-admit prose."""

    bundle = assemble_evidence(record)

    assert command_text(record.tool_calls[1]) is None
    assert bundle.excluded["tool_calls_without_command"] == 1
    assert bundle.excluded["mutations_without_content"] == 1
    assert "storage.py" in "\n".join(bundle.texts)  # the *diff* still names it


# --------------------------------------------------------------------------
# (d) redaction
# --------------------------------------------------------------------------


def test_a_planted_fake_secret_never_reaches_an_item(record: SessionRecord) -> None:
    """The redactor runs before bounding, so caps describe what is sent."""

    bundle = assemble_evidence(record)
    joined = "\n".join(bundle.texts)

    assert FAKE_TOKEN in (record.tool_calls[2].result_text or "")
    assert FAKE_TOKEN not in joined
    assert FAKE_ENV_ASSIGNMENT not in joined
    assert "<redacted>" in joined
    assert "/home/sfx" not in joined  # the home path is rewritten to ~
    assert "~/p/lm/artifacts/attest.json" in joined


def test_a_custom_redactor_is_applied_to_every_item(record: SessionRecord) -> None:
    bundle = assemble_evidence(record, redactor=lambda text: "REPLACED")

    assert set(bundle.texts) == {"REPLACED"}


# --------------------------------------------------------------------------
# (e) bounds and the drop report
# --------------------------------------------------------------------------


def test_bounds_hold_and_the_drop_report_names_what_was_dropped() -> None:
    bundle = assemble_evidence(big_record())
    summary = bundle.summary()

    assert 0 < len(bundle.items) <= EVIDENCE_MAX_ITEMS
    assert all(len(text) <= EVIDENCE_MAX_ITEM_CHARS for text in bundle.texts)
    assert bundle.chars <= EVIDENCE_MAX_TOTAL_CHARS
    assert len("\x1e".join(bundle.texts)) <= EVIDENCE_MAX_TOTAL_CHARS + len(
        bundle.items
    )

    assert summary["caps_hit"], "an over-cap session must say a cap bound it"
    assert summary["dropped"][DIFF_HUNK] > 0
    assert summary["dropped"][COMMAND] > 0
    assert summary["dropped_chars"] > 0
    for name in CATEGORIES:
        assert summary["considered"][name] == summary["kept"][name] + summary["dropped"][name]
    assert sum(summary["kept"].values()) == len(bundle.items)
    assert summary["caps"] == {
        "max_items": EVIDENCE_MAX_ITEMS,
        "max_item_chars": EVIDENCE_MAX_ITEM_CHARS,
        "max_total_chars": EVIDENCE_MAX_TOTAL_CHARS,
    }


def test_a_bounded_session_reports_no_drops(record: SessionRecord) -> None:
    """The report has to be falsifiable in both directions."""

    summary = assemble_evidence(record).summary()

    assert summary["caps_hit"] == []
    assert summary["dropped"] == {name: 0 for name in CATEGORIES}
    assert summary["dropped_chars"] == 0


def test_caller_caps_are_clamped_down_never_up(record: SessionRecord) -> None:
    """A client may submit less than the server accepts, never more."""

    wide = assemble_evidence(record, max_items=10_000, max_total_chars=10**9)
    narrow = assemble_evidence(record, max_items=2)

    assert wide.caps["max_items"] == EVIDENCE_MAX_ITEMS
    assert wide.caps["max_total_chars"] == EVIDENCE_MAX_TOTAL_CHARS
    assert len(narrow.items) == 2
    assert narrow.summary()["caps_hit"] == ["max_items"]


def test_an_oversized_hunk_is_split_not_cut() -> None:
    """Splitting keeps the tail as further items; cutting would lose it."""

    tail = "unique_tail_marker_kept"
    body = "\n".join(f"+    filler_{n} = {n}" for n in range(EVIDENCE_MAX_ITEM_CHARS // 8))
    diff = f"--- a/src/big.py\n+++ b/src/big.py\n@@ -1,0 +1,3 @@\n{body}\n+    {tail} = 1\n"
    record = SessionRecord(
        session_key=SESSION_KEY,
        source="claude",
        cli="claude",
        path=TRANSCRIPT_PATH,
        file_mutations=[
            FileMutation(
                ordinal=0,
                path="src/big.py",
                kind="update",
                span=SourceSpan("claude", TRANSCRIPT_PATH, 1, "toolUseResult"),
                unified_diff=diff,
            )
        ],
    )

    bundle = assemble_evidence(record)

    assert len(bundle.items) > 1
    assert all(len(text) <= EVIDENCE_MAX_ITEM_CHARS for text in bundle.texts)
    assert any(tail in text for text in bundle.texts)
    assert len({item.span.locator() for item in bundle.items}) == len(bundle.items)


def test_identical_items_are_submitted_once() -> None:
    """Two identical writes are one piece of evidence, not two."""

    mutation = dict(
        path="src/dup.py",
        kind="update",
        unified_diff="--- a/src/dup.py\n+++ b/src/dup.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n",
    )
    record = SessionRecord(
        session_key=SESSION_KEY,
        source="claude",
        cli="claude",
        path=TRANSCRIPT_PATH,
        file_mutations=[
            FileMutation(
                ordinal=index,
                span=SourceSpan("claude", TRANSCRIPT_PATH, index, "toolUseResult"),
                **mutation,
            )
            for index in range(3)
        ],
    )

    bundle = assemble_evidence(record)

    assert len(bundle.items) == 1
    assert bundle.summary()["excluded"]["duplicate_items"] == 2


# --------------------------------------------------------------------------
# the shape the runner consumes
# --------------------------------------------------------------------------


def test_hunks_carry_their_own_file_header() -> None:
    hunks = split_diff_hunks(DIFF_TWO_HUNKS)

    assert len(hunks) == 2
    assert all(hunk.startswith("--- a/src/living_memory/storage.py") for hunk in hunks)
    assert "find_recall_attestation" in hunks[1]
    assert "find_recall_attestation" not in hunks[0]


def test_a_deleted_line_that_looks_like_a_header_stays_in_its_hunk() -> None:
    """``--- x`` is what removing a line whose text starts with ``--`` renders.

    The hunk's own line counts are what tell this from a new file header.
    """

    diff = "--- a/n.md\n+++ b/n.md\n@@ -1,2 +1,2 @@\n context\n--- dashes\n+kept_marker\n"

    hunks = split_diff_hunks(diff)

    assert len(hunks) == 1
    assert "kept_marker" in hunks[0]
    assert "--- dashes" in hunks[0]


def test_a_multi_file_diff_splits_at_each_file_header() -> None:
    diff = (
        "--- a/one.py\n+++ b/one.py\n@@ -1,1 +1,1 @@\n-first_old\n+first_new\n"
        "--- a/two.py\n+++ b/two.py\n@@ -9,1 +9,1 @@\n-second_old\n+second_new\n"
    )

    hunks = split_diff_hunks(diff)

    assert len(hunks) == 2
    assert hunks[0].startswith("--- a/one.py\n+++ b/one.py\n@@")
    assert hunks[1].startswith("--- a/two.py\n+++ b/two.py\n@@")
    assert "second_new" not in hunks[0]


def test_a_clipped_diff_keeps_every_hunk_its_counts_no_longer_describe() -> None:
    """Transcripts clip oversized diffs, so headers routinely over-promise."""

    diff = (
        "--- a/big.py\n+++ b/big.py\n"
        "@@ -1,80 +1,80 @@\n-clipped_old\n+clipped_new\n"
        "@@ -900,80 +900,80 @@\n-tail_old\n+tail_new\n"
    )

    hunks = split_diff_hunks(diff)

    assert len(hunks) == 2
    assert "tail_new" in hunks[1]
    assert "tail_new" not in hunks[0]


def test_whole_file_writes_contribute_their_changed_region(
    record: SessionRecord,
) -> None:
    bundle = assemble_evidence(record)
    file_writes = [item for item in bundle.items if item.category == FILE_WRITE]
    joined = "\n".join(item.text for item in file_writes)

    assert len(file_writes) == 2
    # The create has no unchanged part, so all of it is the changed region...
    assert "claim_recall_attestation" in joined
    # ...while the update against a pre-image contributes only what changed.
    assert "beta = 22222" in joined
    assert "gamma" in joined  # context lines survive
    assert not any("alpha = 1\nbeta = 2\ngamma" == item.text for item in file_writes)


def test_proposed_attestations_are_valid_ops(record: SessionRecord) -> None:
    """The runner's op shape, built end to end, must pass ``validate``."""

    bundle = assemble_evidence(record)
    ops = proposed_attestations(record, bundle=bundle)

    assert len(ops) == 1  # the second recall has no captured event id
    op = ops[0]
    assert op.kind == "attest"
    assert op.payload == {
        "recall_event_id": RECALL_EVENT_ID,
        "evidence": list(bundle.texts),
    }
    assert op.provenance.session_key == SESSION_KEY
    assert op.provenance.transcript_sha256 == TRANSCRIPT_SHA
    assert op.provenance.source_span == record.recalls[0].span.locator()
    assert [item.quote for item in op.evidence] == list(bundle.texts)
    assert [item.locator for item in op.evidence] == [
        span.locator() for span in bundle.spans
    ]
    op.validate()


def test_a_session_without_artifacts_proposes_nothing() -> None:
    """No evidence means no op: an op with no evidence is invalid anyway."""

    empty = SessionRecord(
        session_key=SESSION_KEY,
        source="claude",
        cli="claude",
        path=TRANSCRIPT_PATH,
        turns=[Turn(0, "assistant", "I recalled it and thought about it.", None,
                    SourceSpan("claude", TRANSCRIPT_PATH, 0, ""))],
        recalls=[
            RecallInteraction(
                ordinal=0,
                query="anything",
                span=SourceSpan("claude", TRANSCRIPT_PATH, 0, ""),
                recall_event_id=RECALL_EVENT_ID,
            )
        ],
    )

    bundle = assemble_evidence(empty)

    assert bundle.texts == ()
    assert bundle.evidence_sha256 == canonical_evidence(())[1]
    assert proposed_attestations(empty, bundle=bundle) == []


# --------------------------------------------------------------------------
# (f) a real train-split session
# --------------------------------------------------------------------------


def _train_entries() -> list[corpus_mod.SessionEntry]:
    """Sealed ``train`` rows that are safe to read, or skip the test.

    Two filters beyond ``split == "train"``, both about the seal rather than
    about speed:

    * ``group_size == 1`` -- a session's split is bucketed by its *group's*
      union-find root, and a locally rebuilt index sees transcripts recorded
      after the seal. Groups only ever grow, so a session that is alone now was
      alone then, and its split cannot have moved;
    * modified before the manifest's ``generated_at`` -- the session existed
      when the corpus was sealed, so it is a row of the sealed train split and
      not a newcomer the seal never saw.

    Together they make "never a holdout session" hold even against an index
    that is newer than the manifest.
    """

    if not TRACKED_MANIFEST.is_file():
        pytest.skip("no tracked corpus manifest")
    candidates = [
        Path(path)
        for path in (
            os.environ.get(INDEX_ENV),
            corpus_mod.index_path_for(TRACKED_MANIFEST),
            CACHED_INDEX,
        )
        if path
    ]
    index_path = next((path for path in candidates if path.is_file()), None)
    if index_path is None:
        pytest.skip(
            f"no local session index; build one with "
            f"'scripts/postsession_corpus.py --build' or point {INDEX_ENV} at it"
        )
    manifest = corpus_mod.read_manifest(TRACKED_MANIFEST)
    sealed_at = datetime.fromisoformat(
        str(manifest["generated_at"]).replace("Z", "+00:00")
    ).timestamp()
    entries = []
    for entry in corpus_mod.iter_index(index_path):
        if entry.split != "train" or entry.group_size != 1:
            continue
        if not (20_000 < entry.bytes < 4_000_000):
            continue
        path = Path(entry.path)
        try:
            if not path.is_file() or path.stat().st_mtime >= sealed_at:
                continue
        except OSError:  # pragma: no cover - a transcript vanishing mid-run
            continue
        entries.append(entry)
    if not entries:
        pytest.skip("the local index holds no readable sealed train session")
    entries.sort(key=lambda item: item.session_key)
    return entries


def test_a_real_train_session_produces_evidence() -> None:
    """Fixtures prove the rules; a real transcript proves they fire."""

    entries = _train_entries()
    splits_read: set[str] = set()
    produced: list[tuple[corpus_mod.SessionEntry, object]] = []
    for entry in entries[:40]:
        splits_read.add(entry.split)
        record = corpus_mod.load_session(entry)
        bundle = assemble_evidence(record)
        if bundle:
            produced.append((entry, bundle))
        if len(produced) >= 3:
            break

    assert splits_read == {"train"}, "holdout and eval sessions are off limits"
    assert produced, "no real train session produced any evidence"

    for entry, bundle in produced:
        where = entry.session_key
        assert bundle.texts, where
        assert len(bundle.items) <= EVIDENCE_MAX_ITEMS, where
        assert all(len(text) <= EVIDENCE_MAX_ITEM_CHARS for text in bundle.texts), where
        assert bundle.chars <= EVIDENCE_MAX_TOTAL_CHARS, where
        canonical, digest = canonical_evidence(bundle.texts)
        assert canonical == bundle.texts, where
        assert digest == bundle.evidence_sha256, where
        assert {item.category for item in bundle.items} <= set(CATEGORIES), where
        # Reloaded from disk and reassembled: the ledger key must not move.
        reloaded = assemble_evidence(corpus_mod.load_session(entry))
        assert reloaded.evidence_sha256 == bundle.evidence_sha256, where
        assert reloaded.texts == bundle.texts, where
        summary = bundle.summary()
        assert summary["excluded"]["turns"] >= 0
        assert sum(summary["kept"].values()) == len(bundle.items), where


def test_a_real_session_leaks_no_home_path_and_no_memory_result() -> None:
    """Redaction and the memory-tool exclusion, checked on real bytes."""

    entries = _train_entries()
    checked = 0
    for entry in entries[:40]:
        record = corpus_mod.load_session(entry)
        bundle = assemble_evidence(record)
        if not bundle:
            continue
        joined = "\n".join(bundle.texts)
        assert "/home/" not in joined, entry.session_key
        assert "/Users/" not in joined, entry.session_key
        for call in record.tool_calls:
            if not call.is_memory_tool or not call.result_text:
                continue
            body = (call.result_text or "").strip()
            assert body not in joined, entry.session_key
        checked += 1
        if checked >= 3:
            break
    assert checked, "no real train session produced any evidence"
