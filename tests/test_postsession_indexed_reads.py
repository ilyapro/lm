"""Synthetic consumers of byte-identified corpus entries."""

import hashlib
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from living_memory.postsession.corpus import (
    assign_splits, probe, refs_from_index, split_for, write_index,
)
from living_memory.postsession.transcripts import (
    IndexedTranscriptMismatch, TranscriptRef, load_transcript,
)
from scripts import attest_eval, post_session_extract


def codex_line(text):
    body = b'{"type":"event_msg","payload":{"type":"user_message","message":"' + text + b'"}}'
    return body.ljust(81) + b'\n'


@pytest.fixture
def indexed(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_bytes(codex_line(b"first"))
    keys = (f"codex:{i}" for i in range(100))
    key = next(key for key in keys if split_for(key) == "train")
    entry, = assign_splits([probe(TranscriptRef("codex", "codex", path, key))])
    index = tmp_path / "index.jsonl"
    write_index([entry], index)
    return path, entry, index


@pytest.mark.parametrize("change", ["unchanged", "append", "replace", "rewrite_and_grow", "truncate", "missing"])
def test_extract_reads_the_indexed_version(indexed, change):
    path, entry, index = indexed
    original = codex_line(b"first")
    assert entry.bytes == len(original) == 82
    assert entry.sha256 == hashlib.sha256(original).hexdigest()
    changed = codex_line(b"other")
    bodies = {
        "unchanged": original, "append": original + changed, "replace": changed,
        "rewrite_and_grow": changed + original, "truncate": original[:-1],
    }
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(bodies[change])
    args = post_session_extract.build_parser().parse_args(
        ["--split", "train", "--max-bytes", "82", "--index", str(index)]
    )
    messages = []
    records, splits, notes = post_session_extract.select_records(args, messages.append)
    records = list(records)
    assert splits == ["train"]
    if change in {"unchanged", "append"}:
        record, = records
        assert [turn.text for turn in record.turns] == ["first"]
        assert record.transcript_sha256 == entry.sha256
        assert record.turns[0].span.path == str(path)
        assert notes["sessions_failed_to_load"] == 0
    else:
        assert records == []
        assert notes["sessions_failed_to_load"] == 1
        assert entry.session_key in messages[0]
        if change in {"replace", "rewrite_and_grow"}:
            assert "SHA-256 mismatch" in messages[0]


def test_attest_reader_checks_membership_before_open_and_checks_bytes(indexed):
    path, entry, _ = indexed
    corpus = attest_eval.SealedCorpus({}, "", "", "", {entry.session_key: entry},
                                     {"train": frozenset([entry.session_key])}, {})
    reader = attest_eval.TranscriptReader(corpus, "train")
    path.write_bytes(codex_line(b"first") + codex_line(b"later"))
    assert [t.text for t in reader.load(entry).turns] == ["first"]
    path.write_bytes(codex_line(b"other"))
    with pytest.raises(IndexedTranscriptMismatch):
        reader.load(entry)
    path.unlink()
    with pytest.raises(attest_eval.EvalError, match="refusing to open"):
        reader.load(replace(entry, session_key="unsealed"))
    assert reader.opened == {"train": 2}
    assert reader.refused == {"unsealed": 1}


@pytest.mark.parametrize(("source", "original"), [
    ("ae_node_result", b"first"),
    ("ae_chat", b'{"type":"user_message","text":"first"}'),
    ("deepseek", b'{"metadata":{},"messages":[{"role":"user","content":"first"}]}'),
    ("claude", b'{"type":"user","message":{"content":"first"}}'),
    ("codex", codex_line(b"first")),
    ("gigacode", b'{"message":{"role":"user","parts":[{"text":"first"}]}}'),
])
def test_parser_uses_same_bytes_after_source_replacement(tmp_path, monkeypatch, source, original):
    path = tmp_path / "session"
    path.write_bytes(original)
    entry = probe(TranscriptRef(source, source, path, source))
    index = tmp_path / "index"
    write_index(assign_splits([entry]), index)
    ref, = refs_from_index(index)

    # Once the loader has captured its prefix, replace the original in place.
    # Reopening or rewinding the live input for any parser pass sees "other".
    real_open = Path.open

    @contextmanager
    def open_source(self, *args, **kwargs):
        with real_open(self, *args, **kwargs) as handle:
            yield handle
        if self == path:
            with real_open(path, "wb") as writer:
                writer.write(original.replace(b"first", b"other"))

    monkeypatch.setattr(Path, "open", open_source)
    record = load_transcript(ref)
    assert record.transcript_sha256 == entry.sha256
    turn, = record.turns
    assert turn.text == "first"
    assert turn.span.path == record.path == str(path)


def test_large_source_reads_only_indexed_bytes(tmp_path, monkeypatch):
    path = tmp_path / "session"
    original = b"{}\n" * (1 << 20) + codex_line(b"first")
    path.write_bytes(original)
    entry = probe(TranscriptRef("codex", "codex", path, "codex:large"))
    index = tmp_path / "index"
    write_index(assign_splits([entry]), index)
    with path.open("ab") as stream:
        stream.truncate(1 << 32)
    real_open = Path.open
    reads = []

    @contextmanager
    def open_source(self, *args, **kwargs):
        with real_open(self, *args, **kwargs) as stream:
            if self != path:
                yield stream
                return

            class Reader:
                def read(self, size):
                    assert 0 < size <= 1 << 20
                    chunk = stream.read(size)
                    reads.append(len(chunk))
                    assert stream.tell() <= entry.bytes
                    return chunk

            yield Reader()

    monkeypatch.setattr(Path, "open", open_source)
    ref, = refs_from_index(index)
    record = load_transcript(ref)
    assert sum(reads) == entry.bytes
    assert [turn.text for turn in record.turns] == ["first"]
