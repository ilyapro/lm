# Result - goal-instructions-contract-tests

## Summary
The MCP server instruction contract tests now validate the intended behavior of
`_server_instructions(...)` instead of stale literal-keyword and size-window
assertions. `src/living_memory/server.py` was left unchanged.

## Verification command output

### `npm run check`
Worktree: `/home/sfx/p/lm/.worktrees/goal-instructions-contract-tests`

```text
$ npm run check
> lm@0.1.0 check
> bash scripts/check.sh

........................................................................ [ 42%]
........................................................................ [ 84%]
..........................                                               [100%]
170 passed in 20.15s
```

### Instruction contract suite

```text
$ python -m pytest tests/test_instructions_imperative.py -v
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/sfx/p/lm/.worktrees/goal-instructions-contract-tests
configfile: pyproject.toml
plugins: anyio-4.12.1
collected 16 items

tests/test_instructions_imperative.py ................                   [100%]

============================== 16 passed in 0.07s ==============================
```

### Server instruction source diff

```text
$ git diff --stat -- src/living_memory/server.py
(no output)
```

### Current instruction size

```text
$ python -c "from living_memory.server import _server_instructions; t=_server_instructions('global'); print(f'size_bytes={len(t.encode(\"utf-8\"))}'); print(f'size_chars={len(t)}')"
size_bytes=7630
size_chars=7594
```

### Targeted contract inspection

```text
$ grep -nE "MUST NOT|test_must_not_present|MAX_INSTRUCTION_BYTES|test_negative_policy_is_hard_and_concrete|3500|6500" tests/test_instructions_imperative.py
16:MAX_INSTRUCTION_BYTES = 9000
46:def test_negative_policy_is_hard_and_concrete(text: str) -> None:
47:    """Negative policy is semantic, not a literal `MUST NOT` spelling check.
52:    enough, but the exact phrase `MUST NOT` is not required.
185:    assert size <= MAX_INSTRUCTION_BYTES, (
186:        f"instructions size {size} bytes exceeds {MAX_INSTRUCTION_BYTES}; "
```

No stale `3500` or `6500` window assertion remains; no `test_must_not_present`
function remains.

## Contract change
`tests/test_instructions_imperative.py` now treats negative policy as a semantic
contract. The test accepts hard prohibitive language through `MUST NOT`,
`DO NOT`, or `What NOT to store`, then requires concrete evidence: the
`## Anti-patterns` and `## What NOT to store` sections, forbidden storage
categories, and rejection of weak advisory phrases.

The obsolete 3500-6500 byte window is replaced with
`MAX_INSTRUCTION_BYTES = 9000`, a simple explicit upper bound that matches the
current instruction contract while still forcing deliberate review if the
instructions grow further.

The existing behavioral coverage remains in place: recall-before hooks,
remember-after hooks, `memory_teach`, `level:schema`, cross-project recall,
named anti-patterns, weak-language guard, default-scope substitution, and the
cognitive-system opening.

## Living Memory
Fresh `project:lm` trace recorded for this run:
`01KS79A55HYSHRJGEENG9FS57N` (timestamp `2026-05-22T07:27:13Z`).

Prior verification traces this contract chains to:
`01KS78ZG2N3WH45CJ3VYKFGV1B`, `01KS78QDF0AJBS3X6HSNY50SR7`,
`01KS78J59Q1QSMPRAE37WBTG9H`, `01KS78BXEXZEQGDX4SDCNMBNGN`.

The current trace reference is mirrored in
`living_memory_trace_project_lm.json` for artifact-based acceptance checks.
