"""Synthetic correction selection through the detector and default runner."""

from dataclasses import replace
from typing import Any

import pytest

from living_memory.postsession import corrections as C
from living_memory.postsession.judge import FakeJudge, PromptConfig
from living_memory.postsession.session import (
    DeliveredNode, RecallInteraction, SessionRecord, SourceSpan, ToolCall, WriteInteraction,
)

SOURCE = "claude"
PATH = "synthetic/correction-selection.jsonl"
NODE_ID = "01KTVGKXD2W9T9XRHX8X57WB0G"
NODE_CONTENT = (
    "scripts/build_cache.py --build refuses to run when cache_index.json already exists. "
    "The operator must delete the old index manually before rebuilding the cache."
)
OTHER_NODE_CONTENT = (
    "src/retrieval.py ranks bm25 above the vector channel when quoted_query is set, "
    "using the retrieval_weight configuration to order the available results."
)
COMMAND = "python scripts/build_cache.py --build"
OUTPUT = "scripts/build_cache.py --build overwrote cache_index.json in place; scanned 3953 records."
CLAIM = "refuses to run when cache_index.json already exists"
QUOTE = "overwrote cache_index.json in place"
CORRECTION = (
    "scripts/build_cache.py --build overwrites an existing cache_index.json in place "
    "when rebuilding the cache, so the operator does not need to delete the old index manually."
)


def _span(index, field=""):
    return SourceSpan(SOURCE, PATH, index, field)


def make_record(*, nodes=((NODE_ID, NODE_CONTENT),), extra_outputs=(), teach_supersedes=None):
    record = SessionRecord(
        session_key="claude:correction-selection", source=SOURCE, cli="claude", path=PATH,
        cli_session_id="correction-selection", transcript_sha256="a" * 64, record_count=40,
    )
    record.recalls.append(RecallInteraction(
        ordinal=0, query="rebuild the cache", span=_span(1), record_index=1,
        recall_event_id="recall-event-1",
        delivered=[
            DeliveredNode(node_id=node_id, rank=rank, content=content, delivery="full")
            for rank, (node_id, content) in enumerate(nodes)
        ],
    ))
    for ordinal, (command, output, index) in enumerate(((COMMAND, OUTPUT, 10), *extra_outputs)):
        record.tool_calls.append(ToolCall(
            ordinal=ordinal, name="Bash", server=None, arguments={"command": command},
            span=_span(index), result_text=output, ok=True,
        ))
    if teach_supersedes:
        record.writes.append(WriteInteraction(
            ordinal=0, kind="teach", span=_span(20), supersedes=teach_supersedes,
        ))
    return record


def answer(**overrides):
    return {
        "contradicted": True, "reason": "the builder overwrote the existing index",
        "contradicted_claim": CLAIM,
        "observation_locator": f"{SOURCE}:{PATH}#10:tool_calls[0].result_text",
        "observation_quote": QUOTE, "correction": CORRECTION, "confidence": 0.85,
        **overrides,
    }


def repeat_delivery(
    record: SessionRecord, index: int, content: str = NODE_CONTENT,
) -> RecallInteraction:
    recall = replace(
        record.recalls[0],
        ordinal=len(record.recalls),
        span=_span(index, "mcpMeta.structuredContent"),
        record_index=index,
        position=index / record.record_count,
        recall_event_id=f"recall-event-{index}",
        delivered=[
            replace(node, content=content, delivery="full")
            for node in record.recalls[0].delivered
        ],
    )
    record.recalls.append(recall)
    return recall


def judged(record: SessionRecord, *responses: Any) -> tuple[C.DetectionResult, FakeJudge]:
    judge = FakeJudge(responses, prompt_config=PromptConfig(max_payload_bytes=24_000))
    result = C.detect_corrections(record, judge, split="train")
    return result, judge


def only_candidate(record: SessionRecord, node_id: str = NODE_ID) -> C.Candidate:
    for candidate in C.candidates(record):
        if candidate.node.node_id == node_id:
            return candidate
    raise AssertionError(f"no candidate for {node_id}")


@pytest.mark.parametrize(("content", "delivery", "reason"), [
    ("", "below_threshold", "node_content_missing"),
    ("", "full", "node_content_missing"),
    ("scripts/build_cache.py --build", "snippet", "node_content_missing"),
    ("The corpus builder has information about rebuilding an existing index.",
     "snippet", "node_has_no_anchors"),
    (OTHER_NODE_CONTENT, "snippet", "no_related_observation"),
])
def test_detector_selects_later_usable_delivery(content, delivery, reason):
    record = make_record(
        nodes=((NODE_ID, content),), extra_outputs=((COMMAND, OUTPUT, 3),),
    )
    record.recalls[0].delivered[0].delivery = delivery
    selected = repeat_delivery(record, 5)
    candidate = only_candidate(record)
    assert candidate.recall is selected
    assert candidate.node is selected.delivered[0]
    assert [obs.record_index for obs in candidate.observations] == [10]

    result, judge = judged(record, answer())
    assert result.judged == len(judge.calls) == len(result.ops) == 1
    assert result.pair_stats == {reason: 1}
    assert judge.calls[0]["payload"]["stored_fact"]["content"] == NODE_CONTENT
    op = result.ops[0]
    assert op.payload["context"]["recall_event_id"] == selected.recall_event_id
    assert op.payload["context"]["observation_locator"] == answer()["observation_locator"]
    assert op.evidence[1].locator == f"{SOURCE}:{PATH}#5:recalls[1].delivered[0].content"
    assert op.evidence[1].quote == CLAIM


@pytest.mark.parametrize("content", ["", OTHER_NODE_CONTENT])
@pytest.mark.parametrize("index", [10, 15])
def test_detector_cannot_borrow_content_from_a_delivery_after_evidence(content, index):
    record = make_record(nodes=((NODE_ID, content),))
    repeat_delivery(record, index)
    assert C.candidates(record) == []
    result, judge = judged(record, answer())
    assert result.judged == 0
    assert judge.calls == result.ops == []


def test_detector_keeps_first_usable_delivery_and_judges_identity_once():
    record = make_record(nodes=((NODE_ID, ""),))
    selected = repeat_delivery(record, 5)
    selected.delivered[0].delivery = "snippet"
    repeat_delivery(record, 7, NODE_CONTENT + " The scan probes 3953 transcripts.")
    built = C.candidates(record)
    assert len(built) == 1
    assert built[0].recall is selected
    assert built[0].node.content == NODE_CONTENT
    result, judge = judged(record, answer())
    assert result.judged == len(judge.calls) == len(result.ops) == 1
    assert result.pair_stats == {"node_content_missing": 1, "duplicate_delivery": 1}


@pytest.mark.parametrize(("overrides", "reason"), [
    ({"observation_locator": f"{SOURCE}:{PATH}#3:tool_calls[1].result_text"}, "unknown_locator"),
    ({"observation_quote": "The index was overwritten without producing any output."}, "quote_not_verbatim"),
    ({"contradicted_claim": "ranks bm25 above the vector channel"}, "claim_not_in_node"),
])
def test_selected_delivery_still_requires_its_own_claim_and_later_verbatim_evidence(overrides, reason):
    record = make_record(
        nodes=((NODE_ID, OTHER_NODE_CONTENT),), extra_outputs=((COMMAND, OUTPUT, 3),),
    )
    repeat_delivery(record, 5)
    result, judge = judged(record, answer(**overrides))
    assert result.judged == len(judge.calls) == 1
    assert result.ops == []
    assert result.rejections == {reason: 1}


def test_repeated_deliveries_keep_ranking_and_judge_budgets():
    nodes = tuple((f"01NODE{index:020d}", NODE_CONTENT) for index in range(20))
    record = make_record(
        nodes=nodes, extra_outputs=tuple((COMMAND, OUTPUT, i) for i in range(11, 20)),
    )
    repeat_delivery(record, 5)
    built = C.candidates(record)
    cap = C.DEFAULT_CONFIG.max_nodes_per_session
    assert [item.node.node_id for item in built] == [node_id for node_id, _ in nodes[:cap]]
    for candidate in built:
        assert candidate.recall is record.recalls[0]
        assert [obs.record_index for obs in candidate.observations] == [10, 11, 12, 13, 14]
    result, judge = judged(record, *[answer(contradicted=False) for _ in range(cap)])
    assert result.judged == len(judge.calls) == cap
    assert result.pair_stats == {"duplicate_delivery": 20, "over_session_cap": 20 - cap}


def test_runner_proposes_one_correction_from_repeated_delivery(tmp_path):
    from living_memory.postsession.runner import ExtractionRunner, Ledger, RunnerConfig

    record = make_record(nodes=((NODE_ID, ""),))
    repeat_delivery(record, 5)
    repeat_delivery(record, 7)
    judge = FakeJudge({C.TASK: [answer()]})
    runner = ExtractionRunner(
        config=RunnerConfig(), ledger=Ledger(tmp_path / "state"),
        judge_factory=lambda: judge,
    )
    proposals = runner.propose(record)
    assert proposals.errors == []
    assert set(proposals.stage_stats) == {"insights", "corrections", "attest"}
    assert proposals.stage_stats["corrections"]["judged"] == 1
    assert len(proposals.ops) == 1
    assert proposals.ops[0].kind == "teach"
    decisions = runner.decide_and_execute([proposals])
    teach = [decision for decision in decisions if decision.kind == "teach"]
    assert len(teach) == 1
    assert teach[0].status == "would_write"


def test_full_delivery_alone_produces_one_correction():
    record = make_record()
    assert len(C.candidates(record)) == 1
    result, judge = judged(record, answer())
    assert result.judged == len(judge.calls) == len(result.ops) == 1


def test_repeated_deliveries_skip_already_taught_but_recovery_judges_once():
    record = make_record(teach_supersedes=NODE_ID)
    repeat_delivery(record, 5)
    produced, judge = judged(record, answer())
    assert produced.judged == 0
    assert judge.calls == produced.ops == []
    assert produced.pair_stats == {"already_taught_in_session": 2}
    judge = FakeJudge([answer()])
    recovered = C.recover_known_corrections(record, judge)
    assert recovered.judged == len(judge.calls) == len(recovered.ops) == 1
    assert recovered.ops[0].payload["trace_id"] == NODE_ID
