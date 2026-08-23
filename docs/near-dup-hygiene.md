# Near-duplicate hygiene runbook

Collapsing the near-duplicates a corpus has *already* accumulated, with
`scripts/lm_collapse_near_dups.py`. The delivery-side and drain-side collapses
keep new duplication from piling up; this script is the one-off broom for what
piled up before them.

The most recent dry run against a copy of the live database is
[`artifacts/near-dup/hygiene-dryrun.md`](../artifacts/near-dup/hygiene-dryrun.md).

## What it does, and what it refuses to do

* Mean-pools each active node's `node_chunk_embeddings` rows, normalises, and
  compares nodes with nodes.
* Prints the **max-cosine distribution of that database** before applying any
  threshold.
* At the threshold, writes **`supersedes` edges and nothing else**. No row is
  deleted, no `decayed` flag is set, no content is rewritten. Rolling back a pass
  is `DELETE FROM connections WHERE type='supersedes' AND metadata LIKE
  '%lm_collapse_near_dups%'`.
* `--dry-run` is the default and opens the database with `file:...?mode=ro`, so a
  dry run cannot write even by accident.

## Portability: why it is written the way it is

Several Living Memory installations exist, each with its own database, its own
host interpreter, and a checkout at a different commit. So:

* **It imports nothing from `living_memory` and nothing outside the standard
  library.** The contract it relies on is the database schema — `nodes`,
  `node_chunk_embeddings`, `connections` — not the package version. Run it with
  whatever interpreter owns the database (system python on one host, a venv on
  another).
* **numpy is used when it imports and is never required.** On a 13.4k-node corpus
  the numpy scan takes ~1.5 s and the pure-Python scan ~84 s, and the two produce
  the same collapses (see the artifact). `--no-numpy` or `LM_COLLAPSE_NUMPY=0`
  forces the slow path.
* **Schema drift is probed, not assumed.** Optional columns (`level`, `scope`,
  `access_count`, `usefulness_score`, `created_at`) degrade with a printed note;
  `source_traces` is read from the column *and* from a `source_traces` key inside
  `provenance`, because installations differ in which one carries it.

## Preflight refusals

Each one exits 2 with a sentence, rather than failing somewhere inside the scan:

| refusal | meaning |
| --- | --- |
| `node_chunk_embeddings` absent | the database predates chunk embeddings; migrate and backfill first |
| mixed vector widths | vectors of different width are points in different spaces; re-embed before comparing |
| BLOB length ≠ recorded dimensions | the vector store is inconsistent |
| chunk coverage below `--min-coverage` (default 0.50) | a scan over a thin slice would call the corpus deduplicated while most of it was never looked at |
| neither `source_traces` nor `provenance` on `nodes` | the provenance guard cannot be enforced, so nothing may be collapsed |
| threshold below 0.95 without `--allow-mid-band` | see below |
| `connections` absent, with `--apply` | there is nowhere to write the edges |

## The 0.85–0.95 band

Different facts live there. So does the similarity between a concept and the very
traces it was built from — a digest is deliberately byte-different from its
sources, and it lands 0.82–0.95 away from them. Collapsing in that band destroys
provenance and merges distinct facts, so a threshold below 0.95 is refused unless
`--allow-mid-band` is passed after reading that database's own distribution.

## Guards, and the skip reasons they produce

| reason | rule |
| --- | --- |
| `provenance_source_trace` | the node that would be superseded is listed in `source_traces` of a live concept or schema. That band is provenance, never duplication. |
| `level_mismatch` | the two nodes are different levels. A trace is never superseded by a concept written over it. |
| `longer_than_bearer` | the candidate exceeds the bearer by more than `--length-margin` (default 10%). "Same fact plus a new detail" — the detail has to reach the agent. |
| `identifier_veto` | the candidate names an identifier — a ULID, a path, a node or branch name, a slug, a dotted symbol, a digest — that the bearer's text does not. Two texts differing in an identifier are different facts at *any* cosine. |
| `candidate_is_correction` | the candidate is the source of an existing `supersedes` edge; collapsing it would bury the correction. |
| `candidate_already_superseded` | it already has an incoming `supersedes` edge. |
| `bearer_already_superseded` | the node that would bear is itself superseded; nothing is hung off a node the system already demotes. |
| `candidate_already_collapsed` | an earlier, higher-cosine pair in this run already collapsed it. |
| `edge_already_present` | this exact edge exists, so the pass is a no-op for it. |
| `cycle` | the chain would point a node at something it already bears. |

The bearer of a pair is chosen by usage (`access_count`, then
`usefulness_score`, then age, then id). Provenance deliberately does **not** enter
that ranking: a protected node stays a visible skip line rather than being
silently routed around.

Clusters are flattened, not chained: if `LOW → MID` and later `MID → TOP`, both
edges are written against `TOP`, and the length guard **and the identifier veto**
are re-checked against `TOP` before that re-pointing is allowed — what the agent
is left holding is the root's text. A chain dropped by that re-check can leave an
earlier `candidate_already_collapsed` skip line standing for the same node — that
under-collapses, never over-collapses.

### The identifier veto, and its valve

`LM_NEAR_DUP_IDENTIFIER_VETO` is the valve: on unless set to `0`/`false`/`no`/`off`,
and off restores the previous behaviour exactly (no node text is even read). It is
the same env var, with the same default, that `living_memory.near_dup` reads for the
delivery layer and the drain, so one line means one thing everywhere. A dry run
prints its state, counts what it blocked under `identifier_veto`, and — when the
valve is off — carries it into the printed `--apply` command, because a pass run
without the veto is not the pass the default command would apply.

The rule is duplicated in the script rather than imported, which is the price of the
portability above. `tests/test_collapse_near_dups.py` holds it to account: it imports
both implementations — a test may, the script may not — and asserts verdict-for-verdict
agreement over a table covering every identifier class and its negatives, so drift in
either direction fails the suite.

It is deliberately trigger-happy: a false veto costs one uncollapsed near-duplicate,
a false pass hides a fact. On the pre-hygiene backup at 0.95 it blocks **84 of 127**
would-be collapses (127 → 43, 189 pairs skipped), mostly pairs naming different
goal-node paths and different tree branches.

## Running it

```bash
# 1. look. This writes nothing; the database is opened read-only.
python3 scripts/lm_collapse_near_dups.py /path/to/global.sqlite3 --threshold 0.95

# 2. read the distribution section BEFORE deciding on the threshold. The
#    calibration in the artifact is this box's corpus and does not transfer.

# 3. stop the server that owns this database

# 4. apply
python3 scripts/lm_collapse_near_dups.py /path/to/global.sqlite3 --threshold 0.95 --apply

# 5. start the server again
```

**Why the server must be stopped (or restarted right after).** A running server
keeps the scope chunk matrix cached in memory. Collapsing nodes underneath it
leaves that cache stale, so recall keeps serving the nodes the pass just
superseded until the process restarts. The edges also bypass the server's own
supersedes hook (`storage._follow_supersedes_with_anchor_edges`), so query anchor
edges keep pointing at superseded nodes until the next anchor maintenance pass;
recall demotes superseded nodes regardless, so that part is a ranking nicety
rather than a correctness hole.

Applying to the live database is an **operator** step. An agent produces the
dry-run report and the exact command; it does not run the apply, and it does not
stop or restart the server.

## Useful flags

| flag | default | why you would change it |
| --- | --- | --- |
| `--scope-mode same\|any` | `same` | a supersedes edge is a corpus mutation; a `project:x` node should not be superseded by a `project:lm` one that phrases the same sentence. `any` if scopes are known to overlap. |
| `--provenance-levels` | `concept,schema` | both are digests over a cluster, so both protect their sources. `concept` alone is the narrower reading. |
| `--length-margin` | `0.10` | how much longer a candidate may be before it counts as carrying a new detail. |
| `--min-coverage` | `0.50` | lower it deliberately when scanning a partially backfilled database. |
| `--report-limit` | `60` | `0` prints every collapse and every skip. |
| `--json PATH` | — | machine-readable summary: distribution, counts, collapses, skip tallies. |
