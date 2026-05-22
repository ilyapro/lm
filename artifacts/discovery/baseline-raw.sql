-- LM baseline audit SQL.
-- Reproducibility runs against the read-only fixture snapshot at
-- /tmp/lm-baseline-replay.sqlite3 (MD5 58fc12fd47c5f71e5b8867ee71590668),
-- a byte-for-byte copy of /home/sfx/.local/share/living-memory/global.sqlite3
-- taken at 2026-05-22T07:53Z. Inspecting the snapshot under file:...?mode=ro
-- avoids any live external contact and keeps the snapshot MD5 unchanged.
--
-- DB access wrapper used for all queries:
--   python3 - <<'PY'
--   import sqlite3
--   path = '/tmp/lm-baseline-replay.sqlite3'
--   conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
--   conn.row_factory = sqlite3.Row
--   ...
--   PY
-- sqlite3 CLI was unavailable in this environment (`sqlite3: command not found`).

-- Counts by level/decay state.
SELECT level, decayed, COUNT(*) AS count
FROM nodes
GROUP BY level, decayed
ORDER BY level, decayed;

-- Overall duplicate density for active traces.
WITH active_traces AS (
  SELECT content FROM nodes WHERE level = 'trace' AND decayed = 0
)
SELECT
  COUNT(*) AS total_traces,
  COUNT(DISTINCT content) AS distinct_contents,
  COUNT(*) - COUNT(DISTINCT content) AS duplicate_excess,
  ROUND(1.0 * (COUNT(*) - COUNT(DISTINCT content)) / NULLIF(COUNT(*), 0), 6) AS duplicate_density
FROM active_traces;

-- Duplicate density by scope.
WITH active_traces AS (
  SELECT scope, content FROM nodes WHERE level = 'trace' AND decayed = 0
), by_scope AS (
  SELECT
    scope,
    COUNT(*) AS total_traces,
    COUNT(DISTINCT content) AS distinct_contents,
    COUNT(*) - COUNT(DISTINCT content) AS duplicate_excess,
    1.0 * (COUNT(*) - COUNT(DISTINCT content)) / NULLIF(COUNT(*), 0) AS duplicate_density
  FROM active_traces
  GROUP BY scope
)
SELECT scope, total_traces, distinct_contents, duplicate_excess, ROUND(duplicate_density, 6) AS duplicate_density
FROM by_scope
WHERE total_traces >= 10 OR duplicate_excess > 0
ORDER BY duplicate_density DESC, duplicate_excess DESC, total_traces DESC
LIMIT 30;

-- Overall recall_events feedback_applied ratio.
SELECT
  COUNT(*) AS recall_events,
  SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END) AS feedback_applied,
  COUNT(*) - SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END) AS feedback_missing,
  ROUND(1.0 * SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 6) AS feedback_applied_ratio
FROM recall_events;

-- feedback_applied ratio by recall_events.scope.
SELECT
  scope,
  COUNT(*) AS recall_events,
  SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END) AS feedback_applied,
  ROUND(1.0 * SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 6) AS feedback_applied_ratio
FROM recall_events
GROUP BY scope
ORDER BY recall_events DESC
LIMIT 30;

-- Overall never-accessed ratio.
WITH active_nodes AS (
  SELECT level, access_count, last_accessed FROM nodes WHERE decayed = 0
)
SELECT
  'active_nodes' AS population,
  COUNT(*) AS total,
  SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) AS never_accessed,
  ROUND(1.0 * SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 6) AS never_accessed_ratio
FROM active_nodes
UNION ALL
SELECT
  'active_traces' AS population,
  COUNT(*) AS total,
  SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) AS never_accessed,
  ROUND(1.0 * SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 6) AS never_accessed_ratio
FROM nodes WHERE level = 'trace' AND decayed = 0;

-- Never-accessed active traces by scope.
WITH active_traces AS (
  SELECT scope, access_count FROM nodes WHERE level = 'trace' AND decayed = 0
), by_scope AS (
  SELECT
    scope,
    COUNT(*) AS total_traces,
    SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) AS never_accessed,
    1.0 * SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS never_accessed_ratio
  FROM active_traces
  GROUP BY scope
)
SELECT scope, total_traces, never_accessed, ROUND(never_accessed_ratio, 6) AS never_accessed_ratio
FROM by_scope
WHERE total_traces >= 10 OR never_accessed > 0
ORDER BY never_accessed_ratio DESC, never_accessed DESC, total_traces DESC
LIMIT 30;

-- Scope leakage candidate detail.
-- Candidate predicate includes raw Octopus goal-node scopes and project-prefixed
-- recall scopes for rise/*, breakthrough/*, and ocpa-generative-action-substrate-v1.
WITH candidate_scopes AS (
  SELECT scope
  FROM nodes
  WHERE scope = 'rise'
     OR scope LIKE 'rise/%'
     OR scope LIKE 'scope:rise%'
     OR scope = 'breakthrough'
     OR scope LIKE 'breakthrough/%'
     OR scope = 'ocpa-generative-action-substrate-v1'
     OR scope LIKE 'ocpa-generative-action-substrate-v1/%'
     OR scope = 'project:rise'
     OR scope LIKE 'project:rise/%'
     OR scope = 'project:breakthrough'
     OR scope LIKE 'project:breakthrough/%'
     OR scope = 'project:ocpa-generative-action-substrate-v1'
     OR scope LIKE 'project:ocpa-generative-action-substrate-v1/%'
  UNION
  SELECT scope
  FROM recall_events
  WHERE scope = 'rise'
     OR scope LIKE 'rise/%'
     OR scope LIKE 'scope:rise%'
     OR scope = 'breakthrough'
     OR scope LIKE 'breakthrough/%'
     OR scope = 'ocpa-generative-action-substrate-v1'
     OR scope LIKE 'ocpa-generative-action-substrate-v1/%'
     OR scope = 'project:rise'
     OR scope LIKE 'project:rise/%'
     OR scope = 'project:breakthrough'
     OR scope LIKE 'project:breakthrough/%'
     OR scope = 'project:ocpa-generative-action-substrate-v1'
     OR scope LIKE 'project:ocpa-generative-action-substrate-v1/%'
  UNION
  SELECT requested_scope AS scope
  FROM recall_events
  WHERE requested_scope = 'rise'
     OR requested_scope LIKE 'rise/%'
     OR requested_scope LIKE 'scope:rise%'
     OR requested_scope = 'breakthrough'
     OR requested_scope LIKE 'breakthrough/%'
     OR requested_scope = 'ocpa-generative-action-substrate-v1'
     OR requested_scope LIKE 'ocpa-generative-action-substrate-v1/%'
     OR requested_scope = 'project:rise'
     OR requested_scope LIKE 'project:rise/%'
     OR requested_scope = 'project:breakthrough'
     OR requested_scope LIKE 'project:breakthrough/%'
     OR requested_scope = 'project:ocpa-generative-action-substrate-v1'
     OR requested_scope LIKE 'project:ocpa-generative-action-substrate-v1/%'
), node_counts AS (
  SELECT scope,
    COUNT(*) AS total_nodes,
    SUM(CASE WHEN level = 'trace' AND decayed = 0 THEN 1 ELSE 0 END) AS active_traces,
    SUM(CASE WHEN level = 'concept' AND decayed = 0 THEN 1 ELSE 0 END) AS active_concepts,
    SUM(CASE WHEN level = 'schema' AND decayed = 0 THEN 1 ELSE 0 END) AS active_schemas,
    SUM(CASE WHEN decayed = 1 THEN 1 ELSE 0 END) AS decayed_nodes,
    SUM(CASE WHEN decayed = 0 AND access_count = 0 THEN 1 ELSE 0 END) AS active_never_accessed
  FROM nodes
  GROUP BY scope
), recall_scope_counts AS (
  SELECT scope, COUNT(*) AS recall_events_as_scope
  FROM recall_events
  GROUP BY scope
), requested_scope_counts AS (
  SELECT requested_scope AS scope, COUNT(*) AS recall_events_as_requested_scope
  FROM recall_events
  GROUP BY requested_scope
)
SELECT
  c.scope,
  COALESCE(n.total_nodes, 0) AS total_nodes,
  COALESCE(n.active_traces, 0) AS active_traces,
  COALESCE(n.active_concepts, 0) AS active_concepts,
  COALESCE(n.active_schemas, 0) AS active_schemas,
  COALESCE(n.decayed_nodes, 0) AS decayed_nodes,
  COALESCE(n.active_never_accessed, 0) AS active_never_accessed,
  COALESCE(r.recall_events_as_scope, 0) AS recall_events_as_scope,
  COALESCE(req.recall_events_as_requested_scope, 0) AS recall_events_as_requested_scope
FROM candidate_scopes c
LEFT JOIN node_counts n ON n.scope = c.scope
LEFT JOIN recall_scope_counts r ON r.scope = c.scope
LEFT JOIN requested_scope_counts req ON req.scope = c.scope
ORDER BY c.scope;

-- Scope leakage summary using the same candidate predicate.
WITH candidate_nodes AS (
  SELECT * FROM nodes
  WHERE scope = 'rise'
     OR scope LIKE 'rise/%'
     OR scope LIKE 'scope:rise%'
     OR scope = 'breakthrough'
     OR scope LIKE 'breakthrough/%'
     OR scope = 'ocpa-generative-action-substrate-v1'
     OR scope LIKE 'ocpa-generative-action-substrate-v1/%'
     OR scope = 'project:rise'
     OR scope LIKE 'project:rise/%'
     OR scope = 'project:breakthrough'
     OR scope LIKE 'project:breakthrough/%'
     OR scope = 'project:ocpa-generative-action-substrate-v1'
     OR scope LIKE 'project:ocpa-generative-action-substrate-v1/%'
), candidate_scope_details AS (
  WITH candidate_scopes AS (
    SELECT scope FROM candidate_nodes
    UNION
    SELECT scope FROM recall_events
    WHERE scope = 'rise'
       OR scope LIKE 'rise/%'
       OR scope LIKE 'scope:rise%'
       OR scope = 'breakthrough'
       OR scope LIKE 'breakthrough/%'
       OR scope = 'ocpa-generative-action-substrate-v1'
       OR scope LIKE 'ocpa-generative-action-substrate-v1/%'
       OR scope = 'project:rise'
       OR scope LIKE 'project:rise/%'
       OR scope = 'project:breakthrough'
       OR scope LIKE 'project:breakthrough/%'
       OR scope = 'project:ocpa-generative-action-substrate-v1'
       OR scope LIKE 'project:ocpa-generative-action-substrate-v1/%'
    UNION
    SELECT requested_scope AS scope FROM recall_events
    WHERE requested_scope = 'rise'
       OR requested_scope LIKE 'rise/%'
       OR requested_scope LIKE 'scope:rise%'
       OR requested_scope = 'breakthrough'
       OR requested_scope LIKE 'breakthrough/%'
       OR requested_scope = 'ocpa-generative-action-substrate-v1'
       OR requested_scope LIKE 'ocpa-generative-action-substrate-v1/%'
       OR requested_scope = 'project:rise'
       OR requested_scope LIKE 'project:rise/%'
       OR requested_scope = 'project:breakthrough'
       OR requested_scope LIKE 'project:breakthrough/%'
       OR requested_scope = 'project:ocpa-generative-action-substrate-v1'
       OR requested_scope LIKE 'project:ocpa-generative-action-substrate-v1/%'
  ), recall_scope_counts AS (
    SELECT scope, COUNT(*) AS recall_events_as_scope FROM recall_events GROUP BY scope
  ), requested_scope_counts AS (
    SELECT requested_scope AS scope, COUNT(*) AS recall_events_as_requested_scope FROM recall_events GROUP BY requested_scope
  )
  SELECT c.scope,
    COALESCE(r.recall_events_as_scope, 0) AS recall_events_as_scope,
    COALESCE(req.recall_events_as_requested_scope, 0) AS recall_events_as_requested_scope
  FROM candidate_scopes c
  LEFT JOIN recall_scope_counts r ON r.scope = c.scope
  LEFT JOIN requested_scope_counts req ON req.scope = c.scope
)
SELECT
  (SELECT COUNT(DISTINCT scope) FROM candidate_scope_details) AS candidate_scopes,
  (SELECT COUNT(*) FROM candidate_nodes WHERE decayed = 0) AS active_candidate_nodes,
  (SELECT COUNT(*) FROM candidate_nodes WHERE level='trace' AND decayed = 0) AS active_candidate_traces,
  (SELECT SUM(CASE WHEN level='trace' AND decayed=0 AND access_count=0 THEN 1 ELSE 0 END) FROM candidate_nodes) AS active_candidate_traces_never_accessed,
  (SELECT ROUND(1.0 * SUM(CASE WHEN level='trace' AND decayed=0 AND access_count=0 THEN 1 ELSE 0 END) / NULLIF(SUM(CASE WHEN level='trace' AND decayed=0 THEN 1 ELSE 0 END), 0), 6) FROM candidate_nodes) AS active_candidate_trace_never_accessed_ratio,
  (SELECT COUNT(*) FROM candidate_scope_details WHERE recall_events_as_scope = 0 AND recall_events_as_requested_scope = 0) AS candidate_scopes_with_zero_recall_events;

-- Project scope summary used to compare with memory_status coverage.
WITH node_counts AS (
  SELECT scope,
    SUM(CASE WHEN level = 'trace' AND decayed = 0 THEN 1 ELSE 0 END) AS active_traces,
    SUM(CASE WHEN level = 'concept' AND decayed = 0 THEN 1 ELSE 0 END) AS active_concepts,
    SUM(CASE WHEN level = 'schema' AND decayed = 0 THEN 1 ELSE 0 END) AS active_schemas
  FROM nodes
  GROUP BY scope
), recall_counts AS (
  SELECT scope, COUNT(*) AS recall_events
  FROM recall_events
  GROUP BY scope
)
SELECT n.scope, n.active_traces, n.active_concepts, n.active_schemas, COALESCE(r.recall_events, 0) AS recall_events
FROM node_counts n
LEFT JOIN recall_counts r ON r.scope = n.scope
WHERE n.scope IN ('global','project:ae','project:octopus','project:lm','project:online')
ORDER BY n.scope;
