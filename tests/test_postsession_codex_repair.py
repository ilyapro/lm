"""Synthetic Codex rollouts through the ordinary postsession parser and miner."""

import json
from pathlib import Path

import pytest

from living_memory.postsession.evidence import assemble_evidence
from living_memory.postsession.mining import mine
from living_memory.postsession.transcripts import TranscriptRef, load_transcript


def call(key, name="exec_command", **args):
    return {"type": "response_item", "payload": {
        "type": "function_call", "name": name, "call_id": key,
        "arguments": json.dumps(args or {"cmd": "pytest"}),
    }}


def output(key, value):
    return {"type": "response_item", "payload": {
        "type": "function_call_output", "call_id": key, "output": value,
    }}


def event(kind, **fields):
    return {"type": "event_msg", "payload": {"type": kind, **fields}}


def patch(**status):
    return event("patch_apply_end", changes={
        "module.py": {"type": "update", "unified_diff": "@@ -1 +1 @@\n-a\n+b\n"},
    }, **status)


def load(tmp_path: Path, *entries):
    path = tmp_path / "synthetic.jsonl"
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return load_transcript(TranscriptRef("codex", "codex", path, "synthetic"))


RED = "Chunk ID: abc\nWall time: 0.01 seconds\nProcess exited with code 1\nOutput:\nFAILED"
GREEN = "Process exited with code 0\nOutput:\n"
RUNNING = "Process running with session ID 17\nOutput:\n1 passed\nProcess exited with code 0"


@pytest.mark.parametrize(("status", "response", "ok", "changed", "kinds"), [
    ({"success": False}, GREEN, True, False, []),
    ({"success": True}, RUNNING, None, True, []),
    ({"success": True}, GREEN, True, True, ["resolved_failure"]),
    ({"status": "completed"}, GREEN, True, True, ["resolved_failure"]),
    ({"success": True, "status": "declined"}, GREEN, True, False, []),
    ({}, GREEN, True, False, []),
    ({"success": True}, "Output:\n1 passed\nProcess exited with code 0", None, True, []),
    ({"success": True}, "Exit code: 3\nOutput:\n1 passed", False, True, []),
])
def test_red_patch_rerun(tmp_path, status, response, ok, changed, kinds):
    record = load(tmp_path, call("a"), output("a", RED), patch(**status),
                  call("b"), output("b", response))

    assert [c.ok for c in record.tool_calls] == [False, ok]
    assert bool(record.file_mutations) == changed
    assert bool(assemble_evidence(record).kept["diff_hunk"]) == changed
    assert [c.kind for c in mine(record)] == kinds


@pytest.mark.parametrize(("ending", "resolved"), [
    ([call("p", "write_stdin", session_id=17), output("p", [{"type": "input_text", "text": GREEN}])], True),
    ([call("p", "write_stdin", session_id=18), output("p", GREEN)], False),
    ([call("p", "write_stdin", session_id=17), output("wrong", GREEN)], False),
    ([call("p", "write_stdin", session_id=17), output("p", RUNNING)], False),
    ([call("p", "write_stdin", session_id=17), output("p", RED)], False),
    ([event("exec_command_end", call_id="b", process_id="17", exit_code=0, status="completed")], True),
    ([event("exec_command_end", call_id="b", exit_code=0)], True),
    ([event("exec_command_end", call_id="other", process_id="17", exit_code=0)], False),
    ([event("exec_command_end", call_id="b", process_id="18", exit_code=0)], False),
    ([event("exec_command_end", call_id="b", process_id="17", exit_code=False)], False),
])
def test_yielded_command_needs_its_own_completion(tmp_path, ending, resolved):
    record = load(tmp_path, call("a"), output("a", RED), patch(success=True),
                  call("b"), output("b", RUNNING), *ending)
    candidates = mine(record)
    assert [c.kind for c in candidates] == (["resolved_failure"] if resolved else [])
    if resolved:
        green = candidates[0].evidence[-1]
        assert green.quote in ("Process exited with code 0", "0")
        assert f"#{record.record_count - 1}:payload." in green.locator


def test_running_output_after_end_event_keeps_terminal_observation(tmp_path):
    record = load(tmp_path, call("a"), output("a", RED), patch(success=True), call("b"),
                  event("exec_command_end", call_id="b", process_id="17", exit_code=0),
                  output("b", RUNNING))
    green = record.tool_calls[-1]
    assert green.ok is True and green.result_text == RUNNING
    assert mine(record)[0].signals["green_record_index"] == 4


def test_red_completion_after_edit_is_too_late(tmp_path):
    record = load(tmp_path, call("a"), output("a", RUNNING), patch(success=True),
                  event("exec_command_end", call_id="a", process_id="17", exit_code=1),
                  call("b"), output("b", GREEN))
    assert [c.ok for c in record.tool_calls] == [False, True]
    assert mine(record) == []


def test_legacy_shell_metadata_proves_exit_even_without_markers(tmp_path):
    def shell(key):
        return call(key, "shell", command=["pytest"])

    def result(key, code):
        return output(key, json.dumps({"output": "", "metadata": {"exit_code": code}}))

    record = load(tmp_path, shell("a"), result("a", 1), patch(success=True),
                  shell("b"), result("b", 0))
    assert [c.kind for c in mine(record)] == ["resolved_failure"]
