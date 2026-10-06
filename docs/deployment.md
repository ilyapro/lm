# Deployment

Living Memory runs as a long-lived MCP server on two hosts. This page is the
rollout procedure: what is deployed, how to update it, how to restart it, how to
verify the running process actually serves the new code, and how to tell that a
host is behind `master`.

| Host | Updated by | Deployed code | Endpoints |
| --- | --- | --- | --- |
| local dev box | agent or operator | the shared checkout `/home/sfx/p/lm` (editable install) | `http://127.0.0.1:8765/mcp/`, `https://…:8766/mcp/` |
| `alt` (ssh alias) | **the operator, manually** | operator-owned checkout on that host | operator-owned |

**Nothing automated touches `alt`.** No agent connects to it, deploys to it, or
restarts its services. Its connection details live in the operator's
`~/.ssh/config`, not in this repository. The `alt` section below documents the
procedure the operator runs there by hand.

## What is deployed on the local host

Two user systemd units (`~/.config/systemd/user/`):

* `living-memory.service` — the server itself.
  `ExecStart=%h/.local/bin/living-memory-server --transport http --host 127.0.0.1
  --port 8765 --db %h/.local/share/living-memory/global.sqlite3 --default-scope
  global`, with `EnvironmentFile=%h/.config/living-memory/env` (holds
  `LM_AUTH_TOKEN` and the policy variables) and logs appended to
  `~/.local/share/living-memory/server.log`.
* `living-memory-tls.service` — a thin HTTPS `:8766` → HTTP `:8765` bridge. It is
  `PartOf=living-memory.service`, so restarting the server restarts the bridge
  with it; it is not a second server.

`~/.local/bin/living-memory-server` is the console script of an **editable** pip
install:

```sh
python3 -m pip show living-memory | grep -i 'editable project location'
# Editable project location: /home/sfx/p/lm
```

So the code that runs is the working tree at `/home/sfx/p/lm`, read at process
start. That has one consequence that drives everything below: **updating the
deployment is `git pull` in that checkout plus a restart** — there is no copy of
the code to reinstall. Reinstall only when `pyproject.toml` changes
`dependencies` or `[project.scripts]`.

## Tool discovery controls

With no visibility setting, `tools/list` advertises exactly
`memory_recall`, `memory_remember`, `memory_teach`, and `memory_lookup`. The
maintenance tools `memory_consolidate`, `memory_forget`, `memory_connect`,
`memory_status`, and `memory_health` are still registered and directly callable
by name over stdio, HTTP, and HTTPS/TLS; hiding them changes discovery only and
is not an authorization boundary. Normal transport authentication still
applies.

For an operator or offline maintenance stage, put the literal value
`LM_EXPOSE_OPERATOR_TOOLS=1` in the unit's environment file and restart the
server. `tools/list` will then advertise all nine core tools. Removing the
setting (or giving it any value other than `1`) hides the five maintenance
tools again on the next restart; with attestation also off, the result is the
four-tool default. Python embedders have the corresponding
`create_mcp_server(expose_operator_tools=True)` or `False` override: an
explicit boolean wins over the environment, and only `None`/omission reads it.

Attestation is independently default-off. `LM_EXPOSE_ATTEST=1` (or the Python
override `expose_attest=True`) registers and advertises `memory_attest`;
explicit `expose_attest=False` overrides that environment setting. Enabling
attestation does not expose the five maintenance tools, and enabling operator
tools does not register or advertise attestation.

## Update the local host

```sh
git -C /home/sfx/p/lm fetch origin
git -C /home/sfx/p/lm status -sb          # branch, and how far behind origin/master
git -C /home/sfx/p/lm pull --ff-only
systemctl --user restart living-memory.service
```

The running process keeps the old modules in memory until that restart: pulling
without restarting changes nothing about what clients see. `/home/sfx/p/lm` is
also where merges land before they are pushed, so it is often *ahead* of
`origin/master`; `--ff-only` is deliberate — it fails loudly instead of creating
a merge if the checkout and the remote have diverged.

Only when dependencies or entry points changed, re-run the install that created
the current one, then confirm it is still editable and pointed at the same
checkout:

```sh
python3 -m pip install --user -e /home/sfx/p/lm
python3 -m pip show living-memory | grep -i 'editable project location'
```

## Verify the running process

```sh
systemctl --user is-active living-memory.service living-memory-tls.service
curl -fsS http://127.0.0.1:8765/health          # {"ok":true,…,"boot_id":"…"}
python3 /home/sfx/p/lm/scripts/check_deployed_protocol.py
```

`boot_id` changes on every start, so a `boot_id` equal to the one from before
the restart means the restart did not happen. `/health` needs no token and
answers `503` with no `"ok"` in the body while a restart is in flight.

Both `/health` and authorized `/admin/info` include the same boot-time
`code_identity` snapshot. For example, a `/health` response is:

```json
{"ok":true,"service":"living-memory","boot_id":"8f...","code_identity":{"status":"known","scheme":"python-loaded-functions-sha256-v2","digest":"<64 lowercase hex characters>","git_revision":null}}
```

The digest hashes in-memory Python code objects for functions and methods in
LM modules loaded when the server is created, including wrapped functions. It
is fixed for that server instance; changing checkout files or installed
metadata cannot change its response. It is not a Git commit SHA or proof of
the entire package: module-level values, function defaults, later imports,
third-party code and runtime patches are outside this snapshot. If the code
snapshot cannot be identified unambiguously, `status` is `"unknown"` and
`digest` is `null`. `git_revision` is always `null` because the loaded code
cannot be mapped honestly to a Git revision from the running process alone.

The local update Service should read `boot_id` and `code_identity` together
from one response. A changed `boot_id` establishes a new boot, even if its PID
is unchanged after self-exec. For `status: "known"`, compare `scheme` and
`digest` across boots or with a separately launched candidate process; a
matching digest only establishes equality of the covered loaded function code.
Treat `status: "unknown"` or a missing field as inconclusive, and never use
`git_revision: null` as evidence that a desired commit is running.

### Loaded-code serialization migration

Scheme `python-loaded-functions-sha256-v2` serializes each normalized code
object with Python marshal format 0. The earlier v1 used marshal's default
format, whose interned-string tags can depend on benign import and bootstrap
state. For example, recursively interning a nested string constant of the same
function can change v1's bytes without changing its bytecode or constants.
Format 0 removes that distinction for the existing covered function population;
the code still includes nested code objects and still detects changed covered
code. The snapshot remains location independent because code filenames are
normalized before serialization.

To reproduce the migration without a service, database or model, run the narrow
checks from a source checkout:

```sh
PYTHONPATH=src python3 -m pytest -q \
  tests/test_restart_endpoint.py::test_loaded_code_identity_ignores_nested_string_interning_but_detects_changes \
  tests/test_live_task_delivery.py::test_native_projection_accepts_interned_service_and_separate_candidate
```

The first check compares equal code with different string-interning state and
then changes a covered constant. The second starts an isolated service and a
separate candidate process with different benign bootstrap state. A v1 digest
must not be compared with a v2 digest, even when both are 64 hex characters:
capture a prior `boot_id`, `scheme` and `digest` from the same v2 response before
using an observer that requires prior/new loaded-code comparison. The native
task-recall observer explicitly rejects a prior scheme mismatch; do not relabel
an old digest as v2 or treat a cross-scheme difference as code uptake.

Publication of this repair changes `src/living_memory/server.py` itself. An
earlier exact loaded-byte receipt for that file cannot attest the newly
published source. The existing `recall-uses-recorded-task` owner must reconcile
its consumer authority through its supported completion lifecycle against the
normally published repair, including fresh exact-byte evidence for its frozen
31-file loaded-source population (which includes `server.py`). Its earlier
receipt remains historical evidence, not delivery of the changed bytes; its
task behavior and live-reader delivery still need the owner's own verification.
Do not edit that owner's journal, children, input authority or frozen source
list to perform this handoff.

`scripts/check_deployed_protocol.py` is the executable answer to "is this host
serving the code I think it is". It opens an MCP session, reads the server
instructions and the four protocol-bearing tool descriptions (`memory_recall`,
`memory_remember`, `memory_teach`, `memory_lookup`) as the host actually
serves them, and compares them byte-for-byte against the values imported from
the checkout the script runs from:

* exit `0` — the host serves exactly this checkout's protocol texts;
* exit `1` — drift: it prints a unified diff per differing text (and names any
  protocol tool the host does not expose at all);
* exit `2` — the check could not run (connection, auth, or usage error). This is
  never reported as drift.

Those four tools are the default-visible protocol surface, so the checker does
not require operator exposure. In particular, it does not expect hidden
`memory_consolidate` to appear in `tools/list`; maintenance call reachability is
verified separately.

It reads the bearer token from `--token`, else `$LM_AUTH_TOKEN`, else
`LM_AUTH_TOKEN=` in `--env-file` (default `~/.config/living-memory/env`), and
never prints it. Useful variants:

```sh
# the TLS bridge, trusting its self-signed certificate
python3 scripts/check_deployed_protocol.py --tls --port 8766 \
    --ca-cert ~/.config/living-memory/tls/cert.pem

# a host started with a different --default-scope than this checkout's default
python3 scripts/check_deployed_protocol.py --default-scope project:example

# machine-readable verdict (per-text equality, lengths, sha256)
python3 scripts/check_deployed_protocol.py --json
```

The `--default-scope` value only substitutes into the last line of the server
instructions; the local unit runs `--default-scope global`, which is the script's
default. When a host runs a different scope, the report says so explicitly
instead of blaming stale code.

## How to tell a host is behind master

`/admin/info` reports `ok`, `service`, `process_id`, `boot_id`, `started_at`,
`uptime_seconds`, `default_scope`, `argv` and `code_identity` — **no Git code
revision**. The checker below compares the served protocol texts with a
checkout; `code_identity` separately identifies the covered loaded function
code at boot:

1. **Authoritative.** From a checkout at `master` (fetched and up to date), run
   `python3 scripts/check_deployed_protocol.py` against the host. Exit `1` with a
   diff means the host does not serve master's protocol texts — with the
   checkout at master, the host is behind it. Exit `0` means it is current with
   respect to every protocol-bearing text.
2. **Checkout lag (local host).** `git -C /home/sfx/p/lm fetch origin && git -C
   /home/sfx/p/lm status -sb` prints `behind N` when the deployed checkout itself
   has not been pulled.
3. **Process lag (local host).** The process runs whatever the working tree held
   when it started, so a service older than the last source commit is running
   pre-pull code:

   ```sh
   systemctl --user show -p ExecMainStartTimestamp --value living-memory.service
   git -C /home/sfx/p/lm log -1 --format=%cd --date=iso-local -- src/
   ```

Signals 2 and 3 are cheap and catch the common cases (forgot to pull, forgot to
restart); only signal 1 is conclusive, because it compares what the host serves
rather than what its filesystem holds.

## Worktrees are not deployed

The unit serves `/home/sfx/p/lm` — the shared checkout on `master` — and never
`/home/sfx/p/lm/.worktrees/*`. Changing a protocol text in a worktree does not
affect the running service, and restarting the service does not verify worktree
code. To check a worktree's texts, run a separate instance on a spare port with
its own database:

```sh
cd /home/sfx/p/lm/.worktrees/<name>
LM_AUTH_TOKEN=worktree-check PYTHONPATH=src python3 -m living_memory.server \
    --transport http --host 127.0.0.1 --port 8799 \
    --db /tmp/lm-worktree-check.sqlite3 --default-scope global &
python3 scripts/check_deployed_protocol.py --port 8799 --token worktree-check
```

Never point a scratch instance at `~/.local/share/living-memory/global.sqlite3`:
that is the live database of the deployed server.

## The remote host `alt`

`alt` is updated by the operator, by hand, from that host. The steps are the same
three — deliver the commits, make the installed package point at them, restart —
but two things differ from the local host and must be confirmed on `alt` before
the restart:

1. **Commit delivery.** `alt` has no GitHub access from a non-interactive
   session, so `git pull` from `origin` fails there. Deliver commits by
   `git bundle` from the local host (`scp` the bundle, then
   `git pull --ff-only /tmp/<name>.bundle master`), or by pushing to a temporary
   branch in that checkout and fast-forwarding locally on `alt`.
2. **Install mode.** If the install on `alt` is *not* editable (a copy inside a
   virtualenv, e.g. `pip show living-memory` without an
   `Editable project location`), a `git pull` alone changes nothing — the package
   must be reinstalled from the updated checkout
   (`<venv>/bin/pip install --no-deps --force-reinstall <checkout>`) before the
   restart. Reinstalling under a live daemon is safe: the running process holds
   its modules in memory, and the new files take effect at restart.

Restart with whatever unit scope that host uses (`systemctl --user restart
living-memory.service` for a user unit, `sudo systemctl restart
living-memory.service` for a system unit), then verify from `alt` itself exactly
as above:

```sh
curl -fsS http://127.0.0.1:8765/health
python3 <checkout>/scripts/check_deployed_protocol.py --default-scope <that host's scope>
```

The checker compares the host against the checkout it is run from, so run it on
`alt` from `alt`'s checkout after the update. Do not run it against `alt` from
here: reaching that host is the operator's step, not an automated one.
