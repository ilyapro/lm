"""Indexed CLI verification binds each verdict to the stream it opened."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from living_memory.postsession import corpus


@pytest.mark.parametrize("rewrite", [False, True])
def test_verify_follows_open_files_across_replacement(tmp_path, monkeypatch, rewrite):
    roots = corpus.Roots(claude_projects=tmp_path / "projects")
    transcript = roots.claude_projects / "demo" / "00000002-0000-4000-8000-000000000000.jsonl"
    transcript.parent.mkdir(parents=True)
    sealed = b'{"type":"user"}\n'
    transcript.write_bytes(sealed)
    built = corpus.build_corpus(roots)
    entry, = built.entries
    assert entry.split == "holdout"
    manifest, index = corpus.write_corpus(built, tmp_path / "manifest.json")

    # Whitespace, including blank lines and an unterminated row, is sealed verbatim.
    raw = b"\r\n \n " + index.read_bytes().rstrip(b"\n")
    index.write_bytes(raw)
    built.manifest["index_sha256"] = hashlib.sha256(raw).hexdigest()
    corpus.write_manifest(built.manifest, manifest)

    rewritten = sealed.replace(b"user", b"tool")
    observed = (rewritten if rewrite else sealed) + b"!"
    transcript.write_bytes(observed)
    replacements = {index: b"", transcript: sealed if rewrite else rewritten}
    opens = Counter()

    def open_then_replace(path, mode, **kwargs):
        handle = open(path, mode, **kwargs)
        if path in replacements:
            opens[path] += 1
            if opens[path] == 1:
                replacement = path.with_suffix(".replacement")
                replacement.write_bytes(replacements[path])
                replacement.replace(path)
        return handle

    monkeypatch.setattr(corpus, "open", open_then_replace, raising=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "postsession_corpus.py"
    main = runpy.run_path(str(script))["main"]
    report_path = tmp_path / "report.json"
    rc = main(["--verify", "--manifest", str(manifest), "--report", str(report_path)])
    report = json.loads(report_path.read_text())
    assert rc == int(rewrite)
    assert report["ok"] is (not rewrite)
    assert report["cross_check"]["ok"]
    assert report["index"]["actual_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["index"]["actual_lines"] == raw.count(b"\n")
    assert report["checked"] == 1
    assert report["holdout_rewritten" if rewrite else "holdout_appended"] == [entry.session_key]
    change, = report["changed"]
    assert change["actual_sha256"] == hashlib.sha256(observed).hexdigest()
    assert change["actual_bytes"] == len(observed)
    assert opens == {index: 1, transcript: 1}
