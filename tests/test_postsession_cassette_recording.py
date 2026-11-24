"""Offline recording and replay through the extraction entry points."""

from dataclasses import replace
import json
import multiprocessing
from types import SimpleNamespace

import pytest

from living_memory.postsession import runner
from living_memory.postsession.judge import (
    CassetteJudge,
    FakeJudge,
    JudgeCassetteMiss,
    JudgeInvalidOutput,
    JudgeRefused,
)
from postsession_support import make_record
from scripts import post_session_extract as cli
from scripts import postsession_field_run as field


class MeteredJudge(FakeJudge):
    def __init__(self, model):
        self.model = model
        super().__init__({
            "A": [{"answer": "A"}],
            "B": [{"answer": "B"}],
            "C": [JudgeRefused("declined")],
            "D": [JudgeInvalidOutput("invalid answer")],
        })

    def _record(self, stats):
        return super()._record(replace(stats, total_cost_usd=0.25))


def ask(judge, task):
    return judge.judge(task, {"type": "object", "properties": {"answer": {}}}, {}, timeout_s=1)


def test_factories_share_recordings_and_report_actual_spend(tmp_path, monkeypatch):
    path = tmp_path / "cassette-eval.json"
    model = field.EXTRACT_JUDGE_MODEL
    monkeypatch.setattr(cli, "ClaudeCliJudge", MeteredJudge)
    monkeypatch.setattr(field, "ClaudeCliJudge", MeteredJudge)
    factory = cli.make_judge_factory(SimpleNamespace(judge="cassette", cassette=path, model=model))
    first, second = factory(), factory()

    def propose(extractor, records, *, workers):
        third, fourth = extractor.judge_factory(), extractor.judge_factory()
        assert ask(first, "A") == {"answer": "A"}
        assert ask(second, "B") == {"answer": "B"}
        with pytest.raises(JudgeRefused, match="declined"):
            ask(third, "C")
        with pytest.raises(JudgeInvalidOutput, match="invalid answer"):
            ask(fourth, "D")
        assert ask(second, "A") == {"answer": "A"}
        with pytest.raises(JudgeRefused, match="declined"):
            ask(fourth, "C")
        judges = (first, second, third, fourth)
        assert [len(j.inner.calls) for j in judges] == [1] * 4
        assert [j.total_cost_usd for j in judges] == [0.25] * 4
        return [runner.SessionProposals(make_record(), judge_stats=[s for j in judges for s in j.stats])]

    (tmp_path / "preregistration.json").write_text("{}")
    monkeypatch.setattr(field, "DEFAULT_STATE_DIR", tmp_path)
    monkeypatch.setattr(field, "evaluate_gate", lambda *a, **kw: runner.GateDecision("PASS"))
    monkeypatch.setattr(field, "_select_sessions", lambda *a: ([object()], {"eligible": 1}))
    monkeypatch.setattr(runner.ExtractionRunner, "propose_many", propose)
    assert field.cmd_extract(SimpleNamespace(split="eval", state_dir=tmp_path)) == 0
    report = json.loads((tmp_path / "extract-eval.json").read_text())
    assert report["judge"]["calls"] == 6
    assert report["judge"]["total_cost_usd"] == 1.0

    replay = CassetteJudge(path, model=model)
    for task in ("A", "B"):
        assert ask(replay, task) == {"answer": task}
    with pytest.raises(JudgeRefused, match="declined"):
        ask(replay, "C")
    with pytest.raises(JudgeInvalidOutput, match="invalid answer"):
        ask(replay, "D")
    assert all(s.replayed and s.total_cost_usd == 0.25 for s in replay.stats)
    assert replay.total_cost_usd == 0


def _record_worker(path, index, barrier, published, same_key):
    def answer(*args):
        barrier.wait(timeout=10)
        if same_key and index:
            assert published.wait(timeout=10)
        return {"answer": index}

    judge = CassetteJudge(path, inner=FakeJudge([answer]), record=True)
    assert ask(judge, "shared" if same_key else str(index)) == {"answer": index}
    assert not judge.last_stats.replayed
    if index == 0:
        published.set()


@pytest.mark.parametrize("same_key", [False, True])
def test_independent_processes_publish_completed_calls(tmp_path, same_key):
    path = tmp_path / "cassette.json"
    ctx = multiprocessing.get_context("spawn")
    barrier, published = ctx.Barrier(3), ctx.Event()
    workers = [ctx.Process(target=_record_worker, args=(path, i, barrier, published, same_key))
               for i in range(3)]
    for worker in workers:
        worker.start()
    try:
        for worker in workers:
            worker.join(timeout=20)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
            worker.join()

    replay = CassetteJudge(path)
    if same_key:
        assert ask(replay, "shared") == {"answer": 0}
    else:
        for i in range(3):
            assert ask(replay, str(i)) == {"answer": i}


def test_interrupted_replace_preserves_committed_entries(tmp_path, monkeypatch):
    path = tmp_path / "cassette.json"
    judge = CassetteJudge(path, inner=MeteredJudge("sample"), record=True)
    ask(judge, "A")
    original = path.read_bytes()

    def interrupt(*args):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr("os.replace", interrupt)
        with pytest.raises(KeyboardInterrupt):
            ask(judge, "B")
    assert path.read_bytes() == original
    assert judge.total_cost_usd == 0.5
    replay = CassetteJudge(path, model="sample")
    with pytest.raises(JudgeCassetteMiss):
        ask(replay, "B")
    with pytest.raises(JudgeRefused):
        ask(judge, "C")
    assert ask(CassetteJudge(path, model="sample"), "A") == {"answer": "A"}
