import hashlib
from pathlib import Path

import pytest

from living_memory.postsession.corpus import TAIL_BYTES, SessionEntry, hash_and_probe, probe
from living_memory.postsession.transcripts import TranscriptRef


@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("ending", [b"\n", b""])
def test_codex_tail_window(tmp_path: Path, delta: int, ending: bytes) -> None:
    path = tmp_path / "session.jsonl"
    first = b'{"timestamp":"2026-01-01T00:00:00Z"}'
    stamped = b'{"timestamp":"2026-01-01T01:00:00Z"}'
    tail = [stamped]
    tail.extend(b'{"i":%d}' % i for i in range(6))
    tail.append(b"{}")
    padding = TAIL_BYTES + delta - len(b"\n".join(tail)) - len(ending)
    tail[-1] += b" " * padding
    lines = [first, *tail]
    body = b"\n".join(lines) + ending
    path.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()

    ref = TranscriptRef("codex", "codex", path, "codex:sample")
    assert probe(ref) == SessionEntry(
        session_key="codex:sample", source="codex", cli="codex", path=str(path),
        bytes=len(body), sha256=digest, records=9,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T01:00:00Z" if delta <= 0 else None,
    )
    expected_tail = tail if delta <= 0 else tail[1:]
    assert hash_and_probe(path) == (digest, len(body), 9, lines, expected_tail)


@pytest.mark.parametrize("ending", [b"", b"\n", b"\n\n"])
def test_tail_counts_only_nonempty_records(tmp_path: Path, ending: bytes) -> None:
    path = tmp_path / "records"
    lines = [bytes([letter]) for letter in range(ord("a"), ord("k"))]
    path.write_bytes(b"\n\n".join(lines) + ending)
    assert hash_and_probe(path)[-1] == lines[-8:]
