# Example cadence render — do not arm this

Everything under `systemd/` and `cron/` here is an **example render**, produced
by `scripts/v4_cadence.py` and kept in the tree so the exact unit text is
reviewable in a diff. Its paths point at `/nonexistent/EXAMPLE-*` on purpose:
they cannot resolve on any host, so an operator who installs this set by mistake
gets a loud failure from `preflight` rather than a silent missed slot.

`tests/test_v4_cadence.py` asserts this directory is byte-identical to a fresh
render with those example inputs, so it can never drift from the renderer.

To produce a set you could actually arm, render your own:

```sh
python3 -B scripts/v4_cadence.py render \
  --out-dir ~/v4-cadence \
  --repo-root /absolute/path/to/lm \
  --work-dir /absolute/path/outside/the/repo/namespace \
  --campaign-module /absolute/path/to/your/private/campaign.py \
  --cron-timezone "$(cat /etc/timezone)"
```

Then read `docs/v4-campaign-runbook.md` before installing anything. Slot 0 opens
`2026-08-17T00:00:00Z` and closes `2026-08-17T06:00:00Z`; missing any due slot
ends the campaign terminally, with no catch-up, backfill, or replacement.
