# AE contract check: explicit `used` / `irrelevant` fields

Date: 2026-09-27. Read-only review of the AE repo at `/home/sfx/p/ae`. Nothing
in the AE repo was edited.

## Files reviewed

1. `/home/sfx/p/ae/tests/test_mcp_distribution_contract.sh` (317 lines)
2. `/home/sfx/p/ae/tests/e2e/opencode-node/harness/stub/fake_lm.js` (41 lines)

## Findings

### test_mcp_distribution_contract.sh: pins tool NAMES, not schemas

This test writes a synthetic `mcp-servers.json`, runs `sync-mcp`, and checks
the per-client config files it generates. The only Living Memory tool
surface it asserts is the Copilot allowlist:

```python
assert copilot["servers"]["living-memory"]["tools"] == [
    "memory_lookup", "memory_recall", "memory_remember", "memory_teach",
]
assert copilot["servers"]["living-memory"]["deferTools"] == "never"
```

The test never contacts an LM server and never reads `tools/list`,
`inputSchema`, parameter names or descriptions. `grep` finds no
`inputSchema`, `tools/list`, `used`, `irrelevant` or `feedback` in the file.
The optional `codex mcp list --json` step parses only the generated env
references.

**Impact:** none. The change adds no tool and renames none, so the
four-name allowlist stays exact. Optional parameters do not appear in
this contract.

### fake_lm.js: no schema at all

This is the loopback LM double for the offline node E2E. It answers:

- `initialize` with fixed server info;
- every `tools/call` with `{"results": []}`, whatever the tool or arguments;
- everything else (including `tools/list`) with `{}`.

It declares no tools and no input schemas, and it ignores arguments.

**Impact:** none. A client that sends `used`/`irrelevant` gets the same
empty result. The real server adds `feedback_marks` only when the fields are
passed, and this stub is never used as a source of truth for response shape.

## Verdict

Neither file pins LM tool schemas, and **no AE change is required**. The
change is backward compatible from AE's point of view:

- no new tool, so the Copilot `tools` allowlist is unchanged;
- the new parameters are optional (`default: null`, not in `required`), which
  `tests/test_explicit_feedback.py::test_real_fastmcp_schema_carries_optional_fields`
  verifies on the real FastMCP schema;
- responses change only when the caller passes the new fields (a new
  `feedback_marks` key).

AE would need a change only if it later wants its agents to *send* marks,
for example protocol text in AE-generated client configs or a `fake_lm.js`
that echoes `feedback_marks` for an E2E assertion. That is new work, not a
compatibility fix. The rollout handoff should mention it as optional.

A related but separate item: `lm/scripts/check_deployed_protocol.py`
(LM-side, not AE) reports description drift when a host runs
`LM_EXPLICIT_FEEDBACK_PROMPT=mandatory`. See docs/explicit-feedback.md.
