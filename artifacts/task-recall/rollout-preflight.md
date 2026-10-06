# sfx rollout preflight (read only)

Observed 2026-10-06 on sfx, before this retrieval change landed. No checkout, database, unit, or service was changed for this preflight.

## Installed and running state

| Check | Observation |
| --- | --- |
| `python3 -m pip show living-memory` and package `direct_url.json` | Editable install at `/home/sfx/p/lm`; package metadata says `dir_info.editable=true`. The execution worktree is not the installed location. |
| `systemctl --user show living-memory.service` | Active/running, MainPID 274; `ExecStart` uses `~/.local/bin/living-memory-server`, HTTP `127.0.0.1:8765`, the existing live database, config, and `--default-scope global`. Started 2026-10-04 22:24:42 +07. |
| `git -C /home/sfx/p/lm rev-parse HEAD` | `4e0753b8cf5c22bea8fa339d6c16d16c1008bbaf`; clean `master`. This is a checkout revision, not a process attestation. |
| `curl -fsS http://127.0.0.1:8765/health` | `ok=true`; `boot_id=08b2ae3d65d34ac4b6df80fff2025c22`; `code_identity.status=known`, scheme `python-loaded-functions-sha256-v1`, digest `85fabfaf8fca57981c203f07e47cca4125be415048a483c6f080e175928640fd`, `git_revision=null`. These values come from one response. |
| `python3 scripts/check_deployed_protocol.py --json` from this worktree | Exit 1: served server instructions and `memory_recall` description differ from checkout `4e0753b`; the other three protocol tool descriptions match. The checker authenticates through the existing local environment file and compares served MCP texts with imports from this worktree. |

The current service therefore cannot be certified from the editable checkout's HEAD alone. The existing revision-based update status also reports `running=landed=4e0753b`, `stale=false`; that status derives the running revision and does not override the observed served-text mismatch. `/health` is unauthenticated. MCP protocol comparison requires a bearer token; the checker reads it from its configured environment source without printing it. `/admin/info` also requires authorization and is unnecessary for the check above.

## Existing update path and post-landing check

`docs/deployment.md` specifies updating the shared checkout on `master` with a fast-forward pull and restarting `living-memory.service`; an editable install needs no reinstall unless dependency or entry-point metadata changes. The existing local live-code update consumer can perform the same restart for a landed clean revision, but its revision estimate is not loaded-code evidence. Neither path is exercised by this preflight. The retired `alt` host is outside this rollout.

After the retrieval commit is accepted and landed, the delivery owner should:

1. Confirm the target revision is the clean shared `/home/sfx/p/lm` `master` HEAD and the editable install still points there (`python3 -m pip show living-memory`). Record the pre-update `/health` response, including `boot_id` and `code_identity` from the same response.
2. Use the existing local update path in `docs/deployment.md`: fast-forward the shared checkout if needed, then restart `living-memory.service` (or use the existing local live-code consumer for that landed revision). Do not target a worktree or the live database with a scratch process.
3. Confirm `systemctl --user is-active living-memory.service` and poll `curl -fsS http://127.0.0.1:8765/health` until `ok=true`. Require a new `boot_id`. Record the new `code_identity.status`, `scheme`, and `digest`; `unknown` or absent is inconclusive. A digest change demonstrates changed covered in-memory function code, while an unchanged digest can be legitimate if the landed change is outside its coverage. The digest is not a Git revision and excludes module constants, defaults, later imports, and third-party code.
4. Run `python3 /home/sfx/p/lm/scripts/check_deployed_protocol.py --json` against the live service. Require exit 0 for the protocol texts of the landed shared checkout. Exit 1 means actual served-text drift; exit 2 means the comparison could not run, including authentication or connection failure. A matching protocol alone does not prove retrieval internals changed; combine it with the new boot and loaded-code identity, then run the integration node's safe task-recall behavior check.

The delivery owner must perform steps 1–4 after landing. This preflight verifies only the current state and procedure, not uptake of a future commit. No private memory content, memory identifiers, or recall queries are included here.
