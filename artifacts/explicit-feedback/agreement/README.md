# Explicit-feedback agreement check — baselines (2026-09-27)

`scripts/explicit_feedback_agreement.py` scores explicit `used`/`irrelevant`
marks (`recall_feedback_marks`) against placebo-subtracted grounded evidence
plus same-transport lookup, flags ritual marking, and breaks everything down by
agent type. The falsifier test ("better than random") is printed verbatim in
every report.

Baselines were taken before any marks exist (`recall_feedback_marks` is absent
on both stores), so they report **no marks** and the placebo reference rates.

| host | store | window | events (non-A/B) | r1 grounded / twin / excess | top-3 grounded / twin / excess | r1 −q excess | top-3 −q excess |
|---|---|---|---|---|---|---|---|
| sfx | live, `mode=ro` | since 2026-09-07 | 7027 | 8.7% / 2.3% / +6.4 pp | 7.1% / 1.6% / +5.5 pp | +2.5 pp | +2.1 pp |
| alt | snapshot (`sqlite3` backup API over ssh, `mode=ro` source) | since 2026-09-07 | 3170 | 13.9% / 2.6% / +11.3 pp | 11.1% / 2.2% / +8.9 pp | +3.7 pp | +3.2 pp |

(`−q` = query tokens removed from the closing trace.) Hosts are never merged.

Reproduce:

```
PYTHONPATH=src python3 scripts/explicit_feedback_agreement.py \
  --db ~/.local/share/living-memory/global.sqlite3 --host-label sfx --since 2026-09-07 \
  --json artifacts/explicit-feedback/agreement/sfx-baseline.json \
  --md artifacts/explicit-feedback/agreement/sfx-baseline.md
```

and the same with `--db <alt snapshot> --host-label alt`.

Notes:

- The 2026-09-27 chat measurement (LM 01M3H76JBHEZKJMBPWFVN70SB2) gave sfx r1
  12.8% vs 3.7% on 1500 closed events. This run covers all 4837 paired r1 of the
  window with a stricter session-wide A/B filter and gets 8.7% vs 2.3%; the
  ratio (~3.7x) and the cos-bin pattern agree, and the `claude` slice (12.9% vs
  3.3%) matches the earlier figure.
- A/B exclusion here is session-wide and conservative (it drops 5466 sfx /
  1505 alt events). The canonical filter belongs to the effect-daily-metric
  sibling (`scripts/recall_effect_daily.py`).
- Most traffic resolves to agent type `other` (`/root`, `ae:tree_node`, ...),
  because recall events rarely carry an agent. Marks carry their own `agent`
  column, so the per-agent split of marks will be sharper than this reference.
