#!/usr/bin/env python3
"""Post-restart smoke test: recall, then carry a `used` mark on the next call.

Runs against a *running* Living Memory server over HTTP with one MCP client
session, so both calls share one transport session. Steps:

1. ``tools/list`` has exactly the 4 memory tools, and ``memory_recall`` /
   ``memory_remember`` / ``memory_teach`` expose optional ``used`` and
   ``irrelevant`` fields (not in ``required``).
2. ``memory_recall(query)`` returns at least one result.
3. The mark call names the top result as ``used``, plus one id that was never
   delivered. Expect ``feedback_marks = {"accepted": 1, "dropped": 1,
   "by_reason": {"not_delivered": 1}}``. By default the mark rides on a second
   ``memory_recall``, which writes one recall event and two audit rows and no
   node. With ``--remember TEXT`` it rides on a ``memory_remember`` instead,
   which writes TEXT as a trace (scope ``--scope``): use it for the deployment
   note you would store anyway.

This WRITES to the store it talks to (recall events, mark rows, the optional
trace). That is the point of a smoke test, and it is the operator's call.
Under ``LM_EXPLICIT_FEEDBACK_POLICY=audit`` no credit row is claimed.
Under ``off`` the summary is ``ignored: true`` and the script reports it.

    python3 smoke_used_mark.py --env-file ~/.config/living-memory/env
    python3 smoke_used_mark.py --port 18899 --token "$T" --remember "Deploy ..."

The token is read from --token, $LM_AUTH_TOKEN or LM_AUTH_TOKEN= in
--env-file, and it is never printed. Exit 0 on pass, 1 on fail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

TOOLS = {"memory_lookup", "memory_recall", "memory_remember", "memory_teach"}
MARKED_TOOLS = ("memory_recall", "memory_remember", "memory_teach")
FOREIGN_ID = "01SMOKE0NEVER0DELIVERED0000"


def _token(explicit: str | None, env_file: Path | None) -> str | None:
    if explicit:
        return explicit
    if os.environ.get("LM_AUTH_TOKEN"):
        return os.environ["LM_AUTH_TOKEN"]
    if env_file and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("LM_AUTH_TOKEN="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def _payload(result: Any) -> dict[str, Any]:
    data = getattr(result, "structured_content", None) or getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            return json.loads(text)
    raise RuntimeError(f"no JSON payload in tool result: {result!r}")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from fastmcp import Client

    kwargs: dict[str, Any] = {"timeout": args.timeout, "init_timeout": args.timeout}
    token = _token(args.token, args.env_file)
    if token:
        kwargs["auth"] = token
    url = f"http://{args.host}:{args.port}/mcp/"
    report: dict[str, Any] = {"url": url, "checks": {}}
    checks = report["checks"]
    async with Client(url, **kwargs) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        checks["four_tools"] = set(tools) == TOOLS
        optional_fields = {}
        for name in MARKED_TOOLS:
            schema = tools[name].inputSchema if name in tools else {}
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            optional_fields[name] = (
                {"used", "irrelevant"} <= set(props) and not ({"used", "irrelevant"} & required)
            )
        checks["optional_used_irrelevant_fields"] = all(optional_fields.values())
        report["optional_fields"] = optional_fields

        first = _payload(await client.call_tool(
            "memory_recall",
            {"query": args.query, "scope": args.scope, "max_results": 3},
        ))
        results = first.get("results") or []
        checks["recall_returned_results"] = bool(results)
        report["recall_event_id"] = first.get("recall_event_id")
        if not results:
            return report
        top = results[0]["node"]["id"]
        report["marked_used"] = top
        marks = {"used": [top, FOREIGN_ID]}
        if args.remember:
            second = _payload(await client.call_tool(
                "memory_remember",
                {"content": args.remember,
                 "context": {"scope": args.scope, "agent": "rollout-smoke",
                             "task": "explicit-feedback-rollout"},
                 **marks},
            ))
            report["remembered_id"] = (second.get("node") or {}).get("id")
        else:
            second = _payload(await client.call_tool(
                "memory_recall",
                {"query": args.query + " rollout smoke follow-up",
                 "scope": args.scope, "max_results": 1, **marks},
            ))
        summary = second.get("feedback_marks")
        report["feedback_marks"] = summary
        if summary and summary.get("ignored"):
            checks["policy_not_off"] = False
        else:
            checks["used_mark_accepted"] = bool(summary) and summary.get("accepted") == 1
            checks["foreign_id_dropped"] = bool(summary) and summary.get("by_reason") == {
                "not_delivered": 1
            }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--scope", default="project:living-memory")
    parser.add_argument("--query", default="living memory recall credit ledger explicit feedback")
    parser.add_argument("--remember", metavar="TEXT")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    report = asyncio.run(run(args))
    report["ok"] = bool(report["checks"]) and all(report["checks"].values())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
