# Recall-map verification

Focused checks use disposable stores and ordinary consumers:

```sh
LIVING_MEMORY_EMBEDDING_BACKEND=hash python3 -m pytest \
  tests/test_recall_score_gate.py tests/test_recall_map.py \
  tests/test_recall_map_pool.py tests/test_recall_map_delivery.py \
  tests/test_recall_map_failure.py tests/test_recall_map_work.py \
  tests/test_recall_map_curtail.py tests/test_recall_map_anchor_observation.py \
  tests/test_recall_map_skip_pool.py
```

The controls cover current-query qualification, exposure-history swaps, new
knowledge, complementary topics under bounded capacity, cache refresh,
correction inheritance, unchanged main delivery, optional-map failures and
current-work attribution. Scope and correction tests exercise the ordinary MCP
consumer; source-only checks do not establish runtime uptake.

After public source delivery, use the host's ordinary safe LM reload mechanism.
Verify the service's loaded revision and make a useful ordinary recall. Inspect
its main answer and optional map without copying memory contents into source or
another host. Keep reports and delivery receipts outside tracked source.

A focused functional pass establishes the tested behavior. It does not measure
production incidence, aggregate token savings or elapsed-time improvements.
