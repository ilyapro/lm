"""AE chat sidecar observations on synthetic transcripts."""

import builtins
from contextlib import contextmanager
import io
import json
from pathlib import Path

import pytest

from living_memory.postsession.corpus import build_corpus, load_session, probe
from living_memory.postsession.transcripts import ADAPTERS, Roots, TranscriptRef, load_transcript


def chat(root, name, event):
    directory = root / "demo" / "state" / "chats" / name
    directory.mkdir(parents=True)
    path = directory / "transcript.jsonl"
    path.write_text(json.dumps(event) + "\n")
    return path


def metadata(tag):
    return {"model": tag, "workspace": "/" + tag, "session": {"token": tag},
            "sessions_by_provider": {"cli": tag, "other": tag + "-other"}}


def build(root):
    return build_corpus(Roots(ae_projects=root), sources=["ae_chat"], generated_at="fixed")


def test_build_observes_one_version_then_refreshes(tmp_path, monkeypatch):
    path = chat(tmp_path, "one", {"type": "user_message", "text": "hello", "ts": 1000})
    sidecar = path.with_name("meta.json")
    sidecar.write_text(json.dumps(metadata("A")))
    stable_a = build(tmp_path)
    entry, = stable_a.entries
    assert (entry.linked_session_ids, entry.model, entry.cwd) == (("A", "A-other"), "A", "/A")
    original_record = load_session(entry)
    sidecar.write_text(json.dumps(metadata("B")))
    stable_b = build(tmp_path)
    assert stable_a != stable_b

    replacement = sidecar.with_suffix(".new")
    sidecar.rename(replacement)
    sidecar.write_text(json.dumps(metadata("A")))
    reads = []

    def observe(open_file):
        def open_path(file, *args, **kwargs):
            stream = open_file(file, *args, **kwargs)
            if file != sidecar:
                return stream

            @contextmanager
            def observed():
                with stream:
                    class Reader:
                        def read(self, *args):
                            data = stream.read(*args)
                            reads.append(len(data))
                            if replacement.exists():
                                replacement.replace(sidecar)
                            return data

                    yield Reader()

            return observed()
        return open_path

    monkeypatch.setattr(io, "open", observe(io.open))
    monkeypatch.setattr(builtins, "open", observe(builtins.open))
    assert build(tmp_path) == stable_a
    assert reads == [len(json.dumps(metadata("A")))]
    assert build(tmp_path) == stable_b
    assert reads == [len(json.dumps(metadata(tag))) for tag in ("A", "B")]
    with path.open("a") as stream:
        stream.write('{"type":"user_message","text":"later"}\n')
    assert load_session(entry) == original_record
    assert len(reads) == 2


@pytest.mark.parametrize("body,model,cwd,links,warnings", [
    (None, None, None, (), []),
    ("[]", None, None, (), []),
    ("{", None, None, (), ["meta.json unreadable"]),
    ('{"model":[],"workspace":"/ok","sessions_by_provider":{"a":"x","b":4}}',
     None, "/ok", ("x",), []),
    ('{"model":"m","workspace":{},"session":{"token":"y"},"sessions_by_provider":[]}',
     "m", None, ("y",), []),
])
def test_partial_metadata_preserves_both_chats(tmp_path, body, model, cwd, links, warnings):
    path = chat(tmp_path, "one", {"type": "user_message", "text": "hello"})
    if body is not None:
        path.with_name("meta.json").write_text(body)
    other = chat(tmp_path, "two", {"type": "turn_start", "model": "from-turn"})
    other.with_name("meta.json").write_text(json.dumps(metadata("B")))
    first, second = build(tmp_path).entries
    assert (second.model, second.cwd, second.linked_session_ids) == ("from-turn", "/B", ("B", "B-other"))
    assert (first.model, first.cwd, first.linked_session_ids) == (model, cwd, links)
    ref, _ = ADAPTERS["ae_chat"].discover(Roots(ae_projects=tmp_path))
    record = load_transcript(ref)
    assert record.parse_warnings == warnings
    assert [t.text for t in record.turns] == ["hello"]
    assert (record.model, record.cwd, record.linked_session_ids) == (model, cwd, links)
    assert load_session(first).parse_warnings == []


def test_direct_ref_keeps_links_and_reads_current_bounded_sidecar(tmp_path, monkeypatch):
    path = chat(tmp_path, "one", {"type": "user_message", "text": "hello"})
    sidecar = path.with_name("meta.json")
    ref = TranscriptRef("ae_chat", "ae", path, "ae_chat:one", linked_session_ids=("explicit",))
    sidecar.write_bytes(json.dumps(metadata("A")).encode().ljust(4 << 20))
    for result in (probe(ref), load_transcript(ref)):
        assert (result.model, result.cwd, result.linked_session_ids) == ("A", "/A", ("explicit",))
    with sidecar.open("ab") as handle:
        handle.write(b" ")
    assert probe(ref).model is None
    record = load_transcript(ref)
    assert record.cwd is None and record.parse_warnings == []
    sidecar.write_text("{}")
    real_open = Path.open

    def deny_sidecar(self, *args, **kwargs):
        if self == sidecar:
            raise PermissionError
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_sidecar)
    assert probe(ref).linked_session_ids == ("explicit",)
    entry, = build(tmp_path).entries
    assert (entry.model, entry.cwd, entry.linked_session_ids) == (None, None, ())
    assert load_transcript(ref).parse_warnings == ["meta.json unreadable"]
