#!/usr/bin/env python3
"""Remote transcript stream-parser (runs ON alt via `ssh alt python3 -`).

Reads every ~/.claude/projects/*animal-planet*/*.jsonl transcript and emits
one JSON line per living-memory MCP tool_use/tool_result pair, plus per-file
and run summaries. Nothing is written on alt; stdout is redirected to the
local staging file by 40_parse_transcripts.sh.

Record kinds:
- {"kind": "event", ...}         one matched tool_use -> tool_result pair:
    file, line_use, line_result, session_id (transcript sessionId),
    is_sidechain, tool (short name), use_ts, result_ts, query, scope,
    max_results, depth, input_chars, is_error, and THREE serialized-size
    candidates for the tool_result:
      len_text             sum of len(text) over text blocks (or len(str))
      len_json_content     len(json.dumps(content of the tool_result block))
      len_tool_use_result  len(json.dumps(top-level toolUseResult)) or null
- {"kind": "result_size", ...}   one line per tool_result of ANY tool (LM or
    not): file, ts, tool short name, len_json_content + len_text. Powers the
    "LM share of total tool-result volume" cross-checks (18.8% W1 / 3.5% W3)
    and the total-tool-call count cross-check (10669 in W1).
- {"kind": "unmatched_use", ...} tool_use that never got a result in-file
- {"kind": "file_summary", ...}  per file: lines, parse_errors, counts
- {"kind": "run_summary", ...}   totals + tool-name histogram

The query text rides along ONLY so that the local matcher (step 70) can join
events to recall_events rows; it stays inside the private staging dataset.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

GLOB = os.path.expanduser("~/.claude/projects/*animal-planet*/*.jsonl")
LM_MARKERS = ("living-memory", "living_memory")


def is_lm_tool(name: str) -> bool:
    return any(marker in name for marker in LM_MARKERS)


def result_lengths(block: dict, top_level_result) -> dict:
    content = block.get("content")
    if isinstance(content, str):
        len_text = len(content)
    elif isinstance(content, list):
        len_text = sum(
            len(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        len_text = 0
    len_json_content = len(json.dumps(content, ensure_ascii=False)) if content is not None else 0
    len_tool_use_result = (
        len(json.dumps(top_level_result, ensure_ascii=False))
        if top_level_result is not None
        else None
    )
    return {
        "len_text": len_text,
        "len_json_content": len_json_content,
        "len_tool_use_result": len_tool_use_result,
    }


def emit(record: dict) -> None:
    sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    files = sorted(glob.glob(GLOB))
    tool_counter: Counter[str] = Counter()
    total_events = 0
    total_lines = 0
    total_errors = 0

    for path in files:
        rel = os.path.relpath(path, os.path.expanduser("~/.claude/projects"))
        pending: dict[str, dict] = {}
        pending_names: dict[str, str] = {}
        lines = errors = uses = matched = 0
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, 1):
                lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    errors += 1
                    continue
                if not isinstance(obj, dict):
                    continue
                message = obj.get("message") or {}
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                obj_type = obj.get("type")
                if obj_type == "assistant":
                    for block in content:
                        if not isinstance(block, dict) or block.get("type") != "tool_use":
                            continue
                        name = str(block.get("name") or "")
                        pending_names[str(block.get("id"))] = name.rsplit("__", 1)[-1]
                        if not is_lm_tool(name):
                            continue
                        uses += 1
                        tool_counter[name] += 1
                        tool_input = block.get("input") or {}
                        pending[str(block.get("id"))] = {
                            "line_use": line_no,
                            "tool_full": name,
                            "tool": name.rsplit("__", 1)[-1],
                            "use_ts": obj.get("timestamp"),
                            "session_id": obj.get("sessionId"),
                            "is_sidechain": bool(obj.get("isSidechain")),
                            "query": tool_input.get("query"),
                            "scope": tool_input.get("scope"),
                            "max_results": tool_input.get("max_results"),
                            "depth": tool_input.get("depth"),
                            "input_chars": len(json.dumps(tool_input, ensure_ascii=False)),
                        }
                elif obj_type == "user":
                    for block in content:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            continue
                        block_id = str(block.get("tool_use_id"))
                        tool_name = pending_names.pop(block_id, None)
                        if tool_name is not None:
                            sizes = result_lengths(block, None)
                            emit(
                                {
                                    "kind": "result_size",
                                    "file": rel,
                                    "ts": obj.get("timestamp"),
                                    "tool": tool_name,
                                    "len_text": sizes["len_text"],
                                    "len_json_content": sizes["len_json_content"],
                                }
                            )
                        use = pending.pop(block_id, None)
                        if use is None:
                            continue
                        matched += 1
                        total_events += 1
                        record = {
                            "kind": "event",
                            "file": rel,
                            "line_result": line_no,
                            "result_ts": obj.get("timestamp"),
                            "is_error": bool(block.get("is_error")),
                        }
                        record.update(use)
                        record.update(result_lengths(block, obj.get("toolUseResult")))
                        emit(record)
        for use in pending.values():
            emit({"kind": "unmatched_use", "file": rel, **use})
        emit(
            {
                "kind": "file_summary",
                "file": rel,
                "lines": lines,
                "parse_errors": errors,
                "lm_tool_uses": uses,
                "lm_results_matched": matched,
                "unmatched_pending": len(pending),
            }
        )
        total_lines += lines
        total_errors += errors

    emit(
        {
            "kind": "run_summary",
            "files": len(files),
            "total_lines": total_lines,
            "parse_errors": total_errors,
            "events": total_events,
            "tool_histogram": dict(tool_counter.most_common()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
