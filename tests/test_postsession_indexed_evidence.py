"""Evidence from indexed transcripts. To check a local corpus, run

pytest tests/test_postsession_indexed_evidence.py --postsession-corpus MANIFEST INDEX

Use the sealed corpus.json and the index built for that corpus.
"""

from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path

import pytest

from living_memory.attestation import (
    EVIDENCE_MAX_ITEM_CHARS, EVIDENCE_MAX_ITEMS, EVIDENCE_MAX_TOTAL_CHARS,
    canonical_evidence,
)
from living_memory.postsession import corpus
from living_memory.postsession.evidence import CATEGORIES, assemble_evidence
from living_memory.postsession.transcripts import TranscriptRef


def _train_entries(manifest: Path, index: Path) -> list[corpus.SessionEntry]:
    sealed_at = datetime.fromisoformat(corpus.read_manifest(manifest)["generated_at"]).timestamp()
    entries = []
    for entry in corpus.iter_index(index):
        # Only singletons keep their split when the index is rebuilt after the seal.
        if entry.split != "train" or entry.group_size != 1 or not 20_000 < entry.bytes < 4_000_000:
            continue
        path = Path(entry.path)
        try:
            if path.is_file() and path.stat().st_mtime < sealed_at:
                entries.append(entry)
        except OSError:
            continue
    return sorted(entries, key=lambda entry: entry.session_key)


@pytest.fixture
def synthetic_corpus(tmp_path):
    calls = [
        {"type": "tool_use", "id": "cmd", "name": "Bash",
         "input": {"command": "cat /home/example/evidence-fixture.txt"}},
        {"type": "tool_use", "id": "recall", "name": "mcp__living-memory__memory_recall",
         "input": {"query": "fixture"}},
    ]
    result = {"type": "tool_result", "tool_use_id": "recall", "content": "fixture-memory-payload"}
    rows = [
        {"type": "assistant", "message": {"content": calls}},
        {"type": "user", "message": {"content": [result]}},
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows).ljust(20_500) + "\n")
    os.utime(path, (1, 1))
    entry = replace(corpus.probe(TranscriptRef("claude", "claude", path, "claude:train")),
                    split="train", group_size=1)
    manifest = tmp_path / "corpus.json"
    manifest.write_text('{"generated_at":"2000-01-01T00:00:00Z"}')
    index = tmp_path / "corpus-index.jsonl"
    late = tmp_path / "late.jsonl"
    late.touch()
    os.utime(late, (946684800, 946684800))  # exactly at the seal
    corpus.write_index([
        entry,
        replace(entry, session_key="claude:eval", split="eval"),
        replace(entry, session_key="claude:holdout", split="holdout"),
        replace(entry, session_key="claude:group", group_size=2),
        replace(entry, session_key="claude:late", path=str(late)),
        replace(entry, session_key="claude:small", bytes=20_000),
    ], index)
    return manifest, index


def test_only_sealed_singleton_train_entries_are_selected(synthetic_corpus):
    entry, = _train_entries(*synthetic_corpus)
    assert entry.session_key == "claude:train"
    record = corpus.load_session(entry)
    assert any(call.is_memory_tool and call.result_text == "fixture-memory-payload"
               for call in record.tool_calls)
    joined = "\n".join(assemble_evidence(record).texts)
    assert "evidence-fixture.txt" in joined
    assert "/home/" not in joined
    assert "fixture-memory-payload" not in joined


def test_missing_index_is_not_replaced(synthetic_corpus):
    manifest, index = synthetic_corpus
    missing = index.with_name("missing.jsonl")
    with pytest.raises(FileNotFoundError) as error:
        _train_entries(manifest, missing)
    assert error.value.filename == str(missing)


def test_indexed_train_evidence(request):
    selection = request.config.getoption("--postsession-corpus")
    if selection is None:
        selection = request.getfixturevalue("synthetic_corpus")
    entries = _train_entries(*selection)
    checked = 0
    for entry in entries[:40]:
        record = corpus.load_session(entry)
        bundle = assemble_evidence(record)
        if not bundle:
            continue
        where = entry.session_key
        assert bundle.texts, where
        assert len(bundle.items) <= EVIDENCE_MAX_ITEMS, where
        assert all(len(text) <= EVIDENCE_MAX_ITEM_CHARS for text in bundle.texts), where
        assert bundle.chars <= EVIDENCE_MAX_TOTAL_CHARS, where
        assert canonical_evidence(bundle.texts) == (bundle.texts, bundle.evidence_sha256), where
        assert {item.category for item in bundle.items} <= set(CATEGORIES), where
        reloaded = assemble_evidence(corpus.load_session(entry))
        assert reloaded.texts == bundle.texts, where
        assert reloaded.evidence_sha256 == bundle.evidence_sha256, where
        summary = bundle.summary()
        assert summary["excluded"]["turns"] >= 0, where
        assert sum(summary["kept"].values()) == len(bundle.items), where
        joined = "\n".join(bundle.texts)
        assert "/home/" not in joined and "/Users/" not in joined, where
        for call in record.tool_calls:
            if call.is_memory_tool and call.result_text:
                assert call.result_text.strip() not in joined, where
        checked += 1
        if checked == 3:
            break
    assert checked, "no eligible sealed train session produced evidence"
