"""Synthetic consumers of byte-identified corpus entries."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from living_memory.postsession.corpus import (
    assign_splits, probe, refs_from_index, split_for, write_index,
)
from living_memory.postsession.transcripts import (
    ADAPTERS, IndexedTranscriptMismatch, TranscriptRef, load_transcript,
)
from scripts import attest_eval, post_session_extract


def codex_line(text):
    body = b'{"type":"event_msg","payload":{"type":"user_message","message":"' + text + b'"}}'
    return body.ljust(81) + b'\n'


@pytest.fixture
def indexed(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_bytes(codex_line(b"first"))
    key = next(k for i in range(100) if split_for(k := f"codex:{i}") == "train")
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


def test_each_parser_reads_the_verified_snapshot(tmp_path, monkeypatch):
    for source, adapter in ADAPTERS.items():
        path = tmp_path / source
        path.write_bytes(b'{"messages": []}\n')
        entry = probe(TranscriptRef(source, adapter.cli, path, source))
        index = tmp_path / "index"
        write_index(assign_splits([entry]), index)
        ref, = refs_from_index(index)
        parse = adapter._parse

        def replace_before_parse(ref, input_path, sha256):
            path.write_bytes(b'{"messages": [42]}\n')
            assert input_path.read_bytes() == b'{"messages": []}\n'
            return parse(ref, input_path, sha256)

        monkeypatch.setattr(adapter, "_parse", replace_before_parse)
        record = load_transcript(ref)
        assert record.transcript_sha256 == entry.sha256
        assert record.path == str(path)
