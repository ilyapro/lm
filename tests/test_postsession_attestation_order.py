"""Event-specific evidence through the ordinary transcript reader and runner."""

from dataclasses import replace
import json

from living_memory.attestation import EVIDENCE_MAX_ITEMS
from living_memory.postsession.evidence import assemble_evidence, proposed_attestations
from living_memory.postsession.runner import ExtractionRunner, Ledger, RunnerConfig, op_fingerprint
from living_memory.postsession.transcripts import TranscriptRef, load_transcript, sha256_file


def call(call_id, name, **arguments):
    block = {"type": "tool_use", "id": call_id, "name": name, "input": arguments}
    return {"type": "assistant", "message": {"content": [block]}}


def result(call_id, text="", **fields):
    block = {"type": "tool_result", "tool_use_id": call_id, "content": text}
    return {"type": "user", "message": {"content": [block]}, **fields}


def recall(event):
    return call(event, "mcp__living-memory__memory_recall", query="sample")


def delivery(event):
    return result(event, json.dumps({"recall_event_id": event, "results": []}))


def read(tmp_path, rows, source="claude"):
    path = tmp_path / "synthetic.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return load_transcript(TranscriptRef(source, source, path, "synthetic"), sha256=sha256_file(path))


def test_runner_bounds_each_delivery_after_selecting_actions(tmp_path):
    rows = [recall("early")]
    # The only wc call has already finished when its pending recall arrives.
    rows += [call("old", "Bash", command="wc -l rows"), result("old", "37 rows"),
             call("overlap", "Bash", command="probe"), delivery("early"),
             result("overlap", "too late")]
    rows += [call(str(i), "Bash", command=f"sample {i}") for i in range(EVIDENCE_MAX_ITEMS)]
    rows += [recall("late"), call("edit", "Write", file_path="sample"), delivery("late"),
             result("edit", toolUseResult={"filePath": "sample", "content": "middle edit"}),
             call("again", "Bash", command="sample 0"),
             call("new-edit", "Write", file_path="sample"),
             result("new-edit", toolUseResult={"filePath": "sample", "content": "later edit"}),
             call("last", "Bash", command="ls /home/example/data"), result("last", "listed"),
             recall("end"), delivery("end")]
    record = read(tmp_path, rows)
    runner = ExtractionRunner(config=RunnerConfig(), ledger=Ledger(tmp_path / "state"))
    proposals = runner.propose(record)
    assert not proposals.errors
    early, late = proposals.attests
    assert early.payload["recall_event_id"] == "early"
    assert early.payload["evidence"] == ["middle edit", "$ sample 0", "later edit"] + [
        f"$ sample {i}" for i in range(1, EVIDENCE_MAX_ITEMS - 2)
    ]
    assert late.payload == {"recall_event_id": "late", "evidence": [
        "later edit", "$ sample 0", "$ ls ~/data\nlisted",
    ]}
    assert [op_fingerprint(op) for op in runner.propose(record).attests] == [
        op_fingerprint(early), op_fingerprint(late)
    ]
    bundle = assemble_evidence(record, max_items=1, redactor=str.upper)
    assert [op.payload["evidence"] for op in proposed_attestations(record, bundle=bundle)] == [
        ["MIDDLE EDIT"], ["LATER EDIT"],
    ]


def test_uncertain_mutation_and_result_provenance(tmp_path):
    record = read(tmp_path, [
        recall("early"), delivery("early"), call("probe", "Bash", command="probe"),
        result("probe", "from another session"), call("edit", "Write", file_path="file"),
        result("edit", toolUseResult={"filePath": "file", "content": "unlocated edit"}),
    ])
    record.tool_calls[1].result_span = replace(record.span(3), path="other.jsonl")
    record.file_mutations[0].action_span = None
    assert proposed_attestations(record)[0].payload["evidence"] == ["$ probe"]
    record.recalls[0].span = replace(record.span(1), end_record_index=2)
    assert proposed_attestations(record) == []


def test_codex_patch_needs_a_matching_invocation(tmp_path):
    def row(envelope, kind, **fields):
        return {"type": envelope, "payload": {"type": kind, **fields}}

    def patch(call_id):
        return row("response_item", "custom_tool_call", call_id=call_id, name="apply_patch")

    def end(call_id):
        return row("event_msg", "patch_apply_end", call_id=call_id, success=True,
                   changes={"file": {"type": "add", "content": call_id}})

    record = read(tmp_path, [
        patch("before"), row("event_msg", "mcp_tool_call_end",
            invocation={"server": "living-memory", "tool": "memory_recall"},
            result={"Ok": {"structuredContent": {"recall_event_id": "event"}}}),
        end("before"), end("orphan"), patch("after"), end("after"),
        row("event_msg", "mcp_tool_call_end",
            invocation={"tool": "external", "arguments": {"cmd": "unknown start"}}),
    ], source="codex")
    assert proposed_attestations(record)[0].payload["evidence"] == ["after"]


def test_batched_responses_cannot_assign_entry_level_file_content(tmp_path):
    batch = result("old", toolUseResult={"filePath": "old", "content": "earlier edit"})
    batch["message"]["content"] += result("new")["message"]["content"]
    record = read(tmp_path, [
        call("old", "Write", file_path="old"), recall("event"), delivery("event"),
        call("new", "Write", file_path="new"), batch,
    ])
    assert proposed_attestations(record) == []


def test_prepare_only_selected_actions_once_per_proposal(tmp_path, monkeypatch):
    from living_memory.postsession import evidence

    record = read(tmp_path, [
        recall("old"), delivery("old"), call("old-command", "Bash", command="old"),
        recall("one"), delivery("one"), recall("two"), delivery("two"),
        call("write", "Write", file_path="sample"),
        result("write", toolUseResult={"filePath": "sample", "content": "edit"}),
        call("shell", "Bash", command="probe"), result("shell", "output"),
    ])
    prepared = []
    parts = evidence._parts

    def count_parts(text, *args):
        prepared.append(text)
        return parts(text, *args)

    monkeypatch.setattr(evidence, "_parts", count_parts)
    assert proposed_attestations(record, recall_event_ids=[]) == []
    assert prepared == []
    events = ["one", "two"]
    ops = proposed_attestations(record, recall_event_ids=events)
    assert [op.payload for op in ops] == [
        {"recall_event_id": event, "evidence": ["edit", "$ probe\noutput"]} for event in events
    ]
    assert prepared == ["edit", "$ probe\noutput"]
    record.tool_calls[-1].result_text = "updated"
    ops = proposed_attestations(record, recall_event_ids=events)
    assert ops[-1].payload["evidence"][-1] == "$ probe\nupdated"
    assert prepared[2:] == ["edit", "$ probe\nupdated"]


def test_stateful_redactor_sees_each_eligible_body_in_order(tmp_path):
    record = read(tmp_path, [
        recall("one"), delivery("one"), call("first", "Bash", command="first"),
        recall("two"), delivery("two"), call("last", "Bash", command="last"),
    ])
    seen = []

    def redact(text):
        seen.append(text)
        return f"{len(seen)} {text}"

    ops = proposed_attestations(record, redactor=redact)
    assert seen == ["$ first", "$ last", "$ last"]
    assert [op.payload["evidence"] for op in ops] == [
        ["1 $ first", "2 $ last"], ["3 $ last"],
    ]
