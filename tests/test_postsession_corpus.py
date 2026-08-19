"""Corpus enumeration, split assignment and seal tests.

Three properties matter here and all three are checked adversarially rather
than by re-running the implementation:

*The split is the documented function of the key.* The expected bucket is
recomputed from ``sha256(key) % 10`` written out independently in this file,
so a silent change to :func:`~living_memory.postsession.corpus.bucket_for`
fails.

*The seal cannot be moved quietly.* Rewriting, deleting or reclassifying a
holdout transcript has to make ``--verify`` fail, while a session that merely
resumed and appended must not. Neither may the *index* be swapped underneath
the manifest: ``index_sha256`` is what makes the two-file layout as tight as
one file was.

*The tracked artifact stays small and stays metadata.* The first draft of this
module put all 89519 per-session rows in the tracked manifest and breached the
merge gate on its own. So the manifest's size is asserted to be independent of
how many sessions it seals, and the whitelist is asserted to have no room for
rows at all.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from living_memory.postsession.corpus import (  # noqa: E402
    ENTRY_FIELDS,
    EVAL_BUCKETS,
    HOLDOUT_BUCKETS,
    MANIFEST_FIELDS,
    MAX_PROBE_LINE,
    SPLITS,
    TRAIN_BUCKETS,
    Corpus,
    CorpusContentLeak,
    assert_no_content,
    build_corpus,
    hash_and_probe,
    index_path_for,
    load_session,
    read_index,
    read_manifest,
    recovery_stats,
    refs_from_index,
    split_digest,
    verify_corpus,
    write_corpus,
    write_manifest,
)
from living_memory.postsession.transcripts import Roots  # noqa: E402
from test_postsession_transcripts import (  # noqa: E402
    AE_CODEX_SID,
    AGENT_STEM,
    CHAT_ID,
    CLAUDE_SID,
    CODEX_SID,
    DEEP_SID,
    GIGA_SID,
    materialize_fixtures,
)

SCRIPT = ROOT_DIR / "scripts" / "postsession_corpus.py"
TRACKED_MANIFEST = ROOT_DIR / "artifacts" / "post-session" / "corpus.json"
TRACKED_INDEX = ROOT_DIR / "artifacts" / "post-session" / "corpus-index.jsonl"

#: The manifest is the tracked half of a 4000-line merge gate. It carries
#: aggregates only, so its size must not scale with the corpus.
MANIFEST_LINE_BUDGET = 200

#: Every prose sentinel planted in the fixtures. None may reach an artifact.
PROSE_SENTINELS = (
    "REDACTED-QUERY-A",
    "REDACTED-USER-PROMPT",
    "REDACTED-ASSISTANT-TEXT",
    "REDACTED-NODE-CONTENT-1",
    "REDACTED-REMEMBERED-FACT",
    "REDACTED-CORRECTION-TEXT",
    "REDACTED-FILE-BODY",
    "REDACTED-CODEX-USER-MESSAGE",
    "REDACTED-CODEX-REMEMBERED-FACT",
    "REDACTED-ADDED-FILE-BODY",
    "REDACTED-AE-CHAT-PROMPT",
    "REDACTED-NODE-STDOUT-BODY",
    "REDACTED-GIGA-USER-PROMPT",
    "REDACTED-DEEPSEEK-PROMPT",
    "REDACTED-DEEPSEEK-SYSTEM-PROMPT",
    "REDACTED-HOLDOUT-PROMPT-0",
    "REDACTED-HOLDOUT-ANSWER-1",
)

#: Two extra minimal claude transcripts exist only so the fixture corpus has a
#: non-empty holdout to seal, rewrite, delete and append to.
HOLDOUT_FILLERS = (
    "claude:00000002-0000-4000-8000-000000000000",
    "claude:00000004-0000-4000-8000-000000000000",
)


def expected_bucket(key: str) -> int:
    """The documented rule, restated independently of the implementation."""

    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 10


def expected_split(key: str) -> str:
    bucket = expected_bucket(key)
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "eval"
    return "holdout"


@pytest.fixture()
def roots(tmp_path: Path) -> Roots:
    return materialize_fixtures(tmp_path)


@pytest.fixture()
def corpus(roots: Roots) -> Corpus:
    return build_corpus(roots, generated_at="2026-08-19T00:00:00Z")


@pytest.fixture()
def sealed(corpus: Corpus, tmp_path: Path) -> tuple[Path, Path]:
    """A written corpus: the tracked manifest and the ignored index beside it."""

    return write_corpus(corpus, tmp_path / "out" / "corpus.json")


def entry_for(corpus: Corpus, session_key: str) -> dict:
    for entry in corpus.entries:
        if entry.session_key == session_key:
            return entry.to_dict()
    raise AssertionError(f"{session_key!r} not in corpus")


# --------------------------------------------------------------------------
# enumeration
# --------------------------------------------------------------------------


def test_corpus_covers_every_fixture_transcript(corpus: Corpus) -> None:
    keys = {entry.session_key for entry in corpus.entries}
    assert keys == {
        f"claude:{CLAUDE_SID}",
        f"claude:{CLAUDE_SID}:{AGENT_STEM}",
        f"codex:{CODEX_SID}",
        f"codex:{AE_CODEX_SID}",
        f"gigacode:{GIGA_SID}",
        f"deepseek:{DEEP_SID}",
        f"ae_chat:{CHAT_ID}",
        "ae_node:demo:demo-goal",
        *HOLDOUT_FILLERS,
    }
    counts = corpus.manifest["counts"]
    assert counts["sessions"] == 10
    assert counts["by_source"] == {
        "ae_chat": 1,
        "ae_node_result": 1,
        "claude": 4,
        "codex": 2,
        "deepseek": 1,
        "gigacode": 1,
    }
    assert {entry.split for entry in corpus.entries} == set(SPLITS)


def test_index_records_hashes_and_metadata(corpus: Corpus) -> None:
    entry = entry_for(corpus, f"claude:{CLAUDE_SID}")
    path = Path(entry["path"])
    assert path.is_file()
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert entry["bytes"] == path.stat().st_size
    assert entry["records"] == len(path.read_text().splitlines())
    assert entry["cwd"] == "/home/user/p/demo"
    assert entry["repo"] == "/home/user/p/demo"
    assert entry["git_branch"] == "demo-branch"
    assert entry["model"] == "claude-opus-5"
    # the queue-operation record is the earliest stamped line in the file
    assert entry["started_at"] == "2026-08-01T09:59:00.000Z"
    assert entry["ended_at"] is not None

    codex = entry_for(corpus, f"codex:{CODEX_SID}")
    assert codex["git_branch"] == "codex-branch"
    assert codex["model"] == "gpt-5.6-sol"


def test_probe_streams_a_transcript_too_large_to_hold(tmp_path: Path) -> None:
    """``result.md`` reaches 102 MB and one record can be a whole file.

    The probe must still get the right digest, byte count and record count,
    and must not depend on any single record fitting in memory -- so a record
    larger than the probe's own cap is skipped for metadata, not buffered.
    """

    path = tmp_path / "huge.jsonl"
    giant = json.dumps({"type": "file-history-snapshot", "b": "x" * (MAX_PROBE_LINE * 2)})
    body = giant + "\n" + json.dumps({"type": "user", "timestamp": "2026-08-19T00:00:00Z"})
    path.write_text(body, encoding="utf-8")

    digest, size, records, head, tail = hash_and_probe(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert size == path.stat().st_size
    assert records == 2  # the trailing record has no newline and still counts
    assert head[0] == b""  # oversized record dropped rather than buffered
    assert json.loads(head[1])["type"] == "user"
    assert json.loads(tail[-1])["type"] == "user"


def test_corpus_is_deterministic(roots: Roots) -> None:
    first = build_corpus(roots, generated_at="t")
    second = build_corpus(roots, generated_at="t")
    assert json.dumps(first.manifest, sort_keys=True) == json.dumps(
        second.manifest, sort_keys=True
    )
    assert [entry.session_key for entry in first.entries] == sorted(
        entry.session_key for entry in first.entries
    )


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def test_split_follows_the_documented_hash_rule(corpus: Corpus) -> None:
    assert corpus.manifest["split_policy"]["buckets"] == 10
    assert list(TRAIN_BUCKETS) == [0, 1, 2, 3, 4, 5]
    assert list(EVAL_BUCKETS) == [6, 7]
    assert list(HOLDOUT_BUCKETS) == [8, 9]
    for entry in corpus.entries:
        assert entry.bucket == expected_bucket(entry.split_key)
        assert entry.split == expected_split(entry.split_key)


def test_splits_are_disjoint_and_total(corpus: Corpus) -> None:
    buckets: dict[str, set[str]] = {name: set() for name in SPLITS}
    for entry in corpus.entries:
        buckets[entry.split].add(entry.session_key)
    everything = {entry.session_key for entry in corpus.entries}
    assert buckets["train"] | buckets["eval"] | buckets["holdout"] == everything
    assert not buckets["train"] & buckets["eval"]
    assert not buckets["train"] & buckets["holdout"]
    assert not buckets["eval"] & buckets["holdout"]


def test_duplicate_recordings_share_one_split(corpus: Corpus) -> None:
    """A holdout session must not be readable through a train-labelled twin.

    The AE chat, the claude session that served it, and that session's
    subagent transcript are three files recording one conversation; the goal
    node's ``result.md`` and the codex rollout it was printed from are two
    more.
    """

    chat = entry_for(corpus, f"ae_chat:{CHAT_ID}")
    parent = entry_for(corpus, f"claude:{CLAUDE_SID}")
    subagent = entry_for(corpus, f"claude:{CLAUDE_SID}:{AGENT_STEM}")
    assert chat["split_key"] == parent["split_key"] == subagent["split_key"]
    assert chat["split"] == parent["split"] == subagent["split"]
    assert chat["group_size"] == 3

    node_result = entry_for(corpus, "ae_node:demo:demo-goal")
    rollout = entry_for(corpus, f"codex:{AE_CODEX_SID}")
    assert node_result["split_key"] == rollout["split_key"]
    assert node_result["split"] == rollout["split"]
    assert node_result["group_size"] == 2

    # a transcript with no twin keeps its own key, so the documented formula
    # applies to it verbatim
    lonely = entry_for(corpus, f"deepseek:{DEEP_SID}")
    assert lonely["split_key"] == lonely["session_key"]
    assert lonely["group_size"] == 1
    assert corpus.manifest["counts"]["largest_group"] == 3


def test_group_key_survives_the_rolling_claude_window(roots: Roots) -> None:
    """The Claude window prunes at ~30 days; the group's bucket must not move.

    The AE chat outlives the claude transcript it names. Because the split key
    is the union root over the declared ``id:`` identifiers -- not the minimum
    over whichever members happen to still be on disk -- the survivors keep
    the bucket they were sealed with.
    """

    before = build_corpus(roots, generated_at="t")
    chat_before = entry_for(before, f"ae_chat:{CHAT_ID}")

    for entry in before.entries:
        if entry.source == "claude" and entry.session_key.startswith(f"claude:{CLAUDE_SID}"):
            Path(entry.path).unlink()

    after = build_corpus(roots, generated_at="t")
    chat_after = entry_for(after, f"ae_chat:{CHAT_ID}")
    assert chat_after["split_key"] == chat_before["split_key"]
    assert chat_after["bucket"] == chat_before["bucket"]
    assert chat_after["split"] == chat_before["split"]
    assert chat_after["group_size"] == 1


def test_seal_digest_covers_the_sorted_holdout_keys(corpus: Corpus) -> None:
    holdout = sorted(e.session_key for e in corpus.entries if e.split == "holdout")
    body = "".join(f"{key}\n" for key in holdout)
    assert corpus.manifest["holdout_sha256"] == hashlib.sha256(body.encode()).hexdigest()
    assert corpus.manifest["split_sha256"]["holdout"] == corpus.manifest["holdout_sha256"]
    # moving one session in or out changes the seal
    assert split_digest(holdout) != split_digest(holdout + ["claude:intruder"])
    assert holdout and split_digest(holdout) != split_digest(holdout[1:])


# --------------------------------------------------------------------------
# the two-file layout: a small tracked manifest, an ignored index
# --------------------------------------------------------------------------


def test_manifest_has_no_per_session_rows(sealed: tuple[Path, Path]) -> None:
    manifest_path, index_path = sealed
    manifest = read_manifest(manifest_path)
    assert set(manifest) <= set(MANIFEST_FIELDS)
    assert "sessions" not in manifest
    for key, value in manifest.items():
        if isinstance(value, list):
            assert not any(isinstance(item, dict) for item in value), key
    assert index_path.is_file()
    assert len(read_index(index_path)) == manifest["counts"]["sessions"] == 10


def test_manifest_size_does_not_scale_with_the_corpus(roots: Roots, tmp_path: Path) -> None:
    """The defect that blocked the draft, restated as a check.

    Serializing the rows made the tracked artifact 89519 lines. With them in
    the index, sealing one session and sealing ten must cost the same.
    """

    one, _ = write_corpus(
        build_corpus(roots, limit=1, generated_at="t"), tmp_path / "one.json"
    )
    ten, _ = write_corpus(build_corpus(roots, generated_at="t"), tmp_path / "ten.json")
    one_lines = len(one.read_text().splitlines())
    ten_lines = len(ten.read_text().splitlines())
    assert ten_lines < MANIFEST_LINE_BUDGET
    # only the by_split_source / by_cli breakdowns differ
    assert abs(ten_lines - one_lines) <= 20


def test_index_digest_binds_the_manifest_to_the_index(sealed: tuple[Path, Path]) -> None:
    manifest_path, index_path = sealed
    manifest = read_manifest(manifest_path)
    assert manifest["index_sha256"] == hashlib.sha256(index_path.read_bytes()).hexdigest()
    assert manifest["index"]["lines"] == 10
    assert manifest["index"]["fields"] == list(ENTRY_FIELDS)

    ok = verify_corpus(manifest, index_path=index_path)
    assert ok["ok"] is True and ok["index"]["sha256_match"] is True

    rows = index_path.read_bytes().splitlines(keepends=True)
    index_path.write_bytes(b"".join(rows[1:] + rows[:1]))  # same rows, new bytes
    swapped = verify_corpus(manifest, index_path=index_path)
    assert swapped["ok"] is False
    assert swapped["index"]["sha256_match"] is False
    # ...and the rows themselves still cross-check, so the digest is the only
    # thing standing between the manifest and a substituted index
    assert swapped["cross_check"]["ok"] is True


def test_verify_rejects_an_index_whose_rows_were_edited(sealed: tuple[Path, Path]) -> None:
    manifest_path, index_path = sealed
    manifest = read_manifest(manifest_path)
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    victim = next(row for row in rows if row["split"] == "holdout")
    rows = [row for row in rows if row is not victim]
    index_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    report = verify_corpus(manifest, index_path=index_path)
    assert report["ok"] is False
    assert report["index"]["sha256_match"] is False
    assert report["cross_check"]["counts_match"] is False
    assert report["cross_check"]["seal"]["holdout"]["match"] is False


# --------------------------------------------------------------------------
# content invariant
# --------------------------------------------------------------------------


def test_artifacts_carry_no_transcript_content(sealed: tuple[Path, Path]) -> None:
    for path in sealed:
        blob = path.read_text(encoding="utf-8")
        for sentinel in PROSE_SENTINELS:
            assert sentinel not in blob, f"{sentinel} leaked into {path.name}"
    for entry in read_index(sealed[1]):
        assert set(entry.to_dict()) == set(ENTRY_FIELDS)


def test_content_guard_rejects_a_manifest_carrying_rows(corpus: Corpus) -> None:
    """The draft's defect must be unrepresentable, not merely avoided."""

    leaked = json.loads(json.dumps(corpus.manifest))
    leaked["sessions"] = [entry.to_dict() for entry in corpus.entries]
    with pytest.raises(CorpusContentLeak):
        assert_no_content(leaked)


def test_content_guard_rejects_leaks(corpus: Corpus) -> None:
    manifest = corpus.manifest
    rows = [entry.to_dict() for entry in corpus.entries]

    extra = json.loads(json.dumps(rows))
    extra[0]["first_turn"] = "some prose"
    with pytest.raises(CorpusContentLeak):
        assert_no_content(manifest, extra)

    multiline = json.loads(json.dumps(rows))
    multiline[0]["cwd"] = "line one\nline two"
    with pytest.raises(CorpusContentLeak):
        assert_no_content(manifest, multiline)

    huge = json.loads(json.dumps(rows))
    huge[0]["model"] = "x" * 600
    with pytest.raises(CorpusContentLeak):
        assert_no_content(manifest, huge)

    listed = json.loads(json.dumps(rows))
    listed[0]["linked_session_ids"] = ["y" * 600]
    with pytest.raises(CorpusContentLeak):
        assert_no_content(manifest, listed)

    prose = json.loads(json.dumps(manifest))
    prose["generator"] = "line one\nline two"
    with pytest.raises(CorpusContentLeak):
        assert_no_content(prose)


def test_write_refuses_to_persist_a_leak(corpus: Corpus, tmp_path: Path) -> None:
    corpus.manifest["transcript_excerpt"] = "prose"
    with pytest.raises(CorpusContentLeak):
        write_manifest(corpus.manifest, tmp_path / "corpus.json")
    assert not (tmp_path / "corpus.json").exists()

    del corpus.manifest["transcript_excerpt"]
    corpus.entries[0].model = "x" * 600
    with pytest.raises(CorpusContentLeak):
        write_corpus(corpus, tmp_path / "corpus.json")


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def test_verify_passes_on_a_fresh_build(sealed: tuple[Path, Path]) -> None:
    manifest_path, index_path = sealed
    report = verify_corpus(read_manifest(manifest_path), index_path=index_path, strict=True)
    assert report["ok"] is True
    assert report["mode"] == "index"
    assert report["verified"] == report["checked"] == 10
    assert report["changed"] == [] and report["missing"] == []
    assert report["split_mismatches"] == []
    assert all(report["cross_check"]["seal"][name]["match"] for name in SPLITS)


def test_verify_rejects_a_moved_split_label(sealed: tuple[Path, Path]) -> None:
    manifest_path, index_path = sealed
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    victim = next(row for row in rows if row["split"] == "holdout")
    victim["split"] = "train"
    index_path.write_text(
        "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows),
        encoding="utf-8",
    )
    report = verify_corpus(read_manifest(manifest_path), index_path=index_path)
    assert report["ok"] is False
    assert report["split_mismatches"][0]["session_key"] == victim["session_key"]


def test_verify_rejects_a_forged_seal(sealed: tuple[Path, Path]) -> None:
    manifest_path, index_path = sealed
    manifest = read_manifest(manifest_path)
    manifest["holdout_sha256"] = "0" * 64
    report = verify_corpus(manifest, index_path=index_path)
    assert report["ok"] is False
    assert report["cross_check"]["holdout_alias_match"] is False


def test_verify_rejects_a_rewritten_holdout_transcript(sealed: tuple[Path, Path]) -> None:
    manifest_path, index_path = sealed
    victim = next(e for e in read_index(index_path) if e.split == "holdout")
    path = Path(victim.path)
    path.write_text("{}\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    report = verify_corpus(read_manifest(manifest_path), index_path=index_path)
    assert report["ok"] is False
    assert report["holdout_rewritten"] == [victim.session_key]
    assert report["changed"][0]["kind"] == "rewrite"


def test_verify_rejects_a_deleted_holdout_transcript(sealed: tuple[Path, Path]) -> None:
    manifest_path, index_path = sealed
    victim = next(e for e in read_index(index_path) if e.split == "holdout")
    Path(victim.path).unlink()
    report = verify_corpus(read_manifest(manifest_path), index_path=index_path)
    assert report["ok"] is False
    assert report["holdout_missing"] == [victim.session_key]


def test_verify_tolerates_a_resumed_session_appending(sealed: tuple[Path, Path]) -> None:
    """A live session keeps writing to its own transcript; that is not tampering."""

    manifest_path, index_path = sealed
    manifest = read_manifest(manifest_path)
    victim = next(e for e in read_index(index_path) if e.split == "holdout")
    with open(victim.path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "ai-title", "aiTitle": "later"}) + "\n")
    report = verify_corpus(manifest, index_path=index_path)
    assert report["ok"] is True
    assert report["holdout_appended"] == [victim.session_key]
    assert report["holdout_rewritten"] == []
    assert report["changed"][0]["kind"] == "append"
    # ...but --strict still refuses any drift at all
    assert verify_corpus(manifest, index_path=index_path, strict=True)["ok"] is False


def test_verify_reports_train_drift_without_failing(sealed: tuple[Path, Path]) -> None:
    manifest_path, index_path = sealed
    manifest = read_manifest(manifest_path)
    victim = next(e for e in read_index(index_path) if e.split != "holdout")
    Path(victim.path).unlink()
    report = verify_corpus(manifest, index_path=index_path)
    assert report["ok"] is True
    assert victim.session_key in report["missing"]
    assert verify_corpus(manifest, index_path=index_path, strict=True)["ok"] is False


def test_verify_without_an_index_rederives_from_disk(
    sealed: tuple[Path, Path], roots: Roots
) -> None:
    """The index is rebuildable, so losing it must degrade, not break."""

    manifest_path, index_path = sealed
    manifest = read_manifest(manifest_path)
    index_path.unlink()

    clean = verify_corpus(manifest, index_path=index_path, roots=roots, strict=True)
    assert clean["mode"] == "rederive"
    assert clean["ok"] is True
    assert clean["index"]["present"] is False
    assert clean["rederived"]["index_sha256_match"] is True
    assert all(clean["rederived"]["seal"][name]["match"] for name in SPLITS)

    victim = next(e for e in build_corpus(roots).entries if e.split == "holdout")
    Path(victim.path).unlink()
    drifted = verify_corpus(manifest, index_path=index_path, roots=roots)
    assert drifted["ok"] is True, "drift is reported, not failed, without an index"
    assert drifted["rederived"]["counts_drift"]["sessions"]["rederived"] == 9
    assert drifted["rederived"]["seal"]["holdout"]["match"] is False
    assert verify_corpus(manifest, index_path=index_path, roots=roots, strict=True)["ok"] is False


# --------------------------------------------------------------------------
# consumption by downstream children
# --------------------------------------------------------------------------


def test_refs_from_index_filters_by_split(sealed: tuple[Path, Path]) -> None:
    _manifest_path, index_path = sealed
    entries = read_index(index_path)
    assert len(refs_from_index(index_path)) == 10
    for name in SPLITS:
        subset = refs_from_index(index_path, splits=[name])
        expected = {e.session_key for e in entries if e.split == name}
        assert {ref.session_key for ref in subset} == expected


def test_load_session_pins_the_recorded_hash(sealed: tuple[Path, Path]) -> None:
    _manifest_path, index_path = sealed
    entry = next(e for e in read_index(index_path) if e.session_key == f"claude:{CLAUDE_SID}")
    record = load_session(entry)
    assert record.transcript_sha256 == entry.sha256
    assert record.session_key == entry.session_key
    assert len(record.recalls) == 2


def test_recovery_stats_aggregate_the_whole_fixture_corpus(corpus: Corpus) -> None:
    stats = recovery_stats([load_session(entry) for entry in corpus.entries])
    totals = stats["totals"]
    assert totals["sessions"] == 10
    # claude 2 + codex 2 + ae_chat 2 (one id salvaged, one lost) + gigacode 1
    # + deepseek 1
    assert totals["recalls"] == 8
    assert totals["recall_event_ids"] == 7
    assert totals["writes"] == 5
    assert totals["written_nodes"] == 5
    assert totals["file_mutations"] == 5
    assert set(stats["by_source"]) == {
        "ae_chat",
        "ae_node_result",
        "claude",
        "codex",
        "deepseek",
        "gigacode",
    }
    assert 0.0 <= stats["median_position"]["recall"] <= 1.0


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
    )


def cli_args(roots: Roots, target: Path) -> list[str]:
    return [
        "--home",
        str(roots.claude_projects.parent.parent),
        "--ae-root",
        str(roots.ae_projects.parent),
        "--manifest",
        str(target),
    ]


def test_cli_build_verify_stats_round_trip(roots: Roots, tmp_path: Path) -> None:
    target = tmp_path / "out" / "corpus.json"
    common = cli_args(roots, target)

    built = run_cli("--build", *common)
    assert built.returncode == 0, built.stderr
    assert "holdout_sha256=" in built.stdout and "index_sha256=" in built.stdout
    manifest = read_manifest(target)
    assert manifest["counts"]["sessions"] == 10
    assert len(target.read_text().splitlines()) < MANIFEST_LINE_BUDGET
    assert index_path_for(target).is_file()

    verified = run_cli("--verify", "--strict", *common)
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout.strip().endswith("OK")

    stats = run_cli("--stats", "--deep", "--deep-limit", "10", *common)
    assert stats.returncode == 0, stats.stderr
    payload = json.loads(stats.stdout)
    assert payload["counts"]["sessions"] == 10
    assert payload["deep"]["failures"] == []
    # the deep sample defaults to train+eval: the holdout is never parsed
    assert payload["deep"]["splits"] == ["eval", "train"]
    holdout = {e.session_key for e in read_index(index_path_for(target)) if e.split == "holdout"}
    blob = json.dumps(payload)
    assert holdout and not any(key in blob for key in holdout)


def test_cli_deep_stats_honours_the_source_filter(roots: Roots, tmp_path: Path) -> None:
    """`--source` must narrow the deep sample, not just the build enumeration.

    It used to be accepted and ignored here, so `--deep --source deepseek` and
    `--deep --source gigacode` printed byte-identical reports that described
    neither source -- a survey that silently answers a question you did not ask
    is worse than one that refuses.
    """

    target = tmp_path / "corpus.json"
    common = cli_args(roots, target)
    assert run_cli("--build", *common).returncode == 0

    reports = {}
    for source in ("deepseek", "gigacode", "claude"):
        result = run_cli("--stats", "--deep", "--deep-limit", "10", "--source", source, *common)
        assert result.returncode == 0, result.stderr
        deep = json.loads(result.stdout)["deep"]
        assert deep["sources"] == [source]
        assert deep["failures"] == []
        assert deep["parsed"] >= 1
        reports[source] = deep

    # the three reports must actually differ: identical totals were the bug
    assert reports["deepseek"]["totals"] != reports["gigacode"]["totals"]
    assert {s for s in reports["deepseek"]["by_source"]} == {"deepseek"}
    assert {s for s in reports["gigacode"]["by_source"]} == {"gigacode"}
    assert {s for s in reports["claude"]["by_source"]} == {"claude"}


def test_cli_verify_fails_when_the_seal_moved(roots: Roots, tmp_path: Path) -> None:
    target = tmp_path / "corpus.json"
    common = cli_args(roots, target)
    assert run_cli("--build", *common).returncode == 0
    manifest = read_manifest(target)
    manifest["holdout_sha256"] = "f" * 64
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    failed = run_cli("--verify", *common)
    assert failed.returncode == 1
    assert "FAILED" in failed.stdout


def test_cli_verify_reports_a_missing_manifest() -> None:
    assert run_cli("--verify", "--manifest", "/nonexistent/corpus.json").returncode == 2


# --------------------------------------------------------------------------
# the tracked artifact
# --------------------------------------------------------------------------


def test_tracked_manifest_is_a_valid_sealed_corpus() -> None:
    """The acceptance artifact must exist, be non-empty and carry no content."""

    assert TRACKED_MANIFEST.is_file(), f"missing required artifact {TRACKED_MANIFEST}"
    manifest = read_manifest(TRACKED_MANIFEST)
    assert_no_content(manifest)
    assert manifest["counts"]["sessions"] > 0
    assert "sessions" not in manifest
    assert len(manifest["index_sha256"]) == 64
    assert manifest["holdout_sha256"] == manifest["split_sha256"]["holdout"]
    # every source of the real corpus is represented, and no split is empty
    assert set(manifest["counts"]["by_source"]) == {
        "ae_chat",
        "ae_node_result",
        "claude",
        "codex",
        "deepseek",
        "gigacode",
    }
    assert all(manifest["counts"]["by_split"][name] > 0 for name in SPLITS)
    assert sum(manifest["counts"]["by_split"].values()) == manifest["counts"]["sessions"]


def test_tracked_manifest_stays_under_the_merge_gate() -> None:
    """The draft shipped 89519 lines here and blocked its own merge."""

    lines = len(TRACKED_MANIFEST.read_text(encoding="utf-8").splitlines())
    assert lines < MANIFEST_LINE_BUDGET, f"corpus.json is {lines} lines"


def test_the_index_is_ignored_and_the_manifest_is_not() -> None:
    """The rows are rebuildable and full of absolute home paths: never tracked."""

    if not (ROOT_DIR / ".git").exists():
        # Already running from an export, which is the proof this test wants.
        pytest.skip("not a git checkout")

    def tracked(path: Path) -> bool:
        return bool(
            subprocess.run(
                ["git", "-C", str(ROOT_DIR), "ls-files", "--", str(path.relative_to(ROOT_DIR))],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )

    def ignored(path: Path) -> bool:
        return (
            subprocess.run(
                ["git", "-C", str(ROOT_DIR), "check-ignore", "-q", "--no-index",
                 str(path.relative_to(ROOT_DIR))],
                capture_output=True,
            ).returncode
            == 0
        )

    assert tracked(TRACKED_MANIFEST), "the seal must be committed"
    assert not ignored(TRACKED_MANIFEST)
    assert ignored(TRACKED_INDEX), ".gitignore must exclude the per-session index"
    assert not tracked(TRACKED_INDEX)


def test_tracked_manifest_matches_its_index_when_one_is_present() -> None:
    """On a machine that has built the corpus, the seal must still hold."""

    if not TRACKED_INDEX.is_file():
        pytest.skip("no local index; run scripts/postsession_corpus.py --build")
    manifest = read_manifest(TRACKED_MANIFEST)
    assert manifest["index_sha256"] == hashlib.sha256(TRACKED_INDEX.read_bytes()).hexdigest()
    entries = read_index(TRACKED_INDEX)
    assert len(entries) == manifest["counts"]["sessions"]
    for entry in entries:
        assert entry.sha256 and entry.bytes > 0
        assert entry.bucket == expected_bucket(entry.split_key)
        assert entry.split == expected_split(entry.split_key)
    assert manifest["holdout_sha256"] == split_digest(
        entry.session_key for entry in entries if entry.split == "holdout"
    )
