# Field verification of the recall-map relevance selector

This is the operator's runbook for the fresh-field verification of the frozen
`directional-zsum-r1` selector on the two hosts `sfx` and `alt`. It is the
executable form of the sealed preseal
`artifacts/recall-map/relevance/field/protocol.json`
(`protocol_id: directional-zsum-r1-fresh-field-20260823`). That file is the
authority; this page adds no threshold, no floor and no instrument of its own.
Where the two ever disagree, the protocol wins and the disagreement is itself a
protocol conflict — stop and report it.

Everything below is run **by the operator, on the named host, after this branch
is merged**. No agent performs any of it.

| Constant | Value |
| --- | --- |
| Protocol id / namespace | `directional-zsum-r1-fresh-field-20260823` |
| Policy id | `directional-zsum-r1` |
| Candidate implementation commit | `5b7f4c0c3ba94b1d4d36fc0b9fcb00a6e8bcdf64` |
| Lineage anchor commit | `24db9bbb33f30e507a1b58af7eedc6163bc962ab` |
| `CARRIER:src` git tree | `f45eeb5a17375340c6f3a7dc6ec7674f96b251df` |
| `CARRIER:src/living_memory` git tree | `0181c33c6d3d0763beb856176a9f0cdb155a5b6c` |
| Deployed-package typed-content sha256 | `b302ede43343532dddda437c12c8b3ed7a54d07a18b9c67248dfe9944b1eb4ce` |
| `policy.json` bytes / sha256 | 16815 / `1f065b9e0e86b415ba8f7ea94649c9590b8e1f36aa1f312eb12d9afe6f9afe2b` |
| `selected_policy` sha256 (= `recall_map.py::RELEVANCE_POLICY_DIGEST`) | `3acad3d92db2538bf3096ab99d2c4d337ea3c8aebddc226646527b4d277660ea` |
| Stable source bytes / sha256 | 514121728 / `9f00abf4467b96bfb6f3cdce9123edfc275a392030da6833e3f7b58b3dcbabcc` |
| Sealed organic rate / required rate | 0.4662 / 0.233 |
| Pinned warm pooled p95 overhead | 0.1958 (budget 0.20) |

## Scope and execution boundary

**Everything in this document runs outside the repository worktree, on `sfx` or
on `alt`, after the goal branch is merged.** Nothing here is a condition of
accepting the branch. The branch ships the selector, the frozen policy, the
sealed instruments and this runbook; the field verdict is produced afterwards.

**No pre-existing host state is evidence.** A store, seed, snapshot, journal
archive or receipt already sitting on either host from an earlier attempt is
*not* evidence, however plausible its path looks. For each such artifact do
exactly one of two things:

1. Verify it against the invariants in [Fresh-store invariants](#fresh-store-invariants)
   and the acceptance rule of its schema in [Receipt schemas](#receipt-schemas) —
   byte length, SHA-256, mode `0444`, producer host, protocol hash, production
   instant — and only then treat it as an input; or
2. Move it aside (never delete it if it may bear on an earlier protocol
   conflict) and rebuild it from the stable source named below.

A receipt that cannot be verified is a missing receipt. Report it missing.
Never complete, re-date, re-sign, re-hash or reconstruct one off-host.

**Where the field files live.** `policy.json` `holdout_boundary` forward-refers
to `artifacts/recall-map/relevance/field/manifest.json#candidate_holdout`, and
`protocol.json` `tracked_outputs` lists six repository paths. Five of the six do
not exist on this branch and are not created by it. They are produced by the
operator, on `master` after the merge, in this exact order and nowhere else:

| Path | Produced when | By |
| --- | --- | --- |
| `artifacts/recall-map/relevance/field/protocol.json` | already committed; the operator commits later additive revisions | operator |
| `artifacts/recall-map/relevance/field/manifest.json` | after both accrual receipts validate, from the outcome-blind inventory only | operator |
| `artifacts/recall-map/relevance/field/map-effect.json` | after the embargo is released | operator |
| `artifacts/recall-map/relevance/field/injection-sfx.json` | after the embargo is released | operator |
| `artifacts/recall-map/relevance/field/injection-alt.json` | after the embargo is released | operator |
| `artifacts/recall-map/relevance/field/verdict.md` | last | operator |

Those six are the **only** repository-tracked outputs of this subtree. Stores,
seeds, snapshots, journal archives, promotion evidence, helper scripts and every
preparation, promotion, deployment, accrual and rollback receipt are external
inputs and must never be copied under the repository worktree.

**Disclosed carrier-contract deviation.** `deployment_carrier_contract`
`.exact_nonruntime_allowlist` is closed at exactly two runtime-neutral blobs
(`scripts/render_recall_map_relevance_report.py`,
`tests/test_recall_map_relevance_report.py`) and its required check rejects "any
third non-field path" in the carrier diff. This runbook is a third non-runtime
path. It is registered additively by the new top-level `protocol.json` key
`external_execution.runbook`; the sealed allowlist, thresholds, `tracked_outputs`
and the embargo are untouched. So when checking the carrier, the expected diff
against `5b7f4c0c3ba94b1d4d36fc0b9fcb00a6e8bcdf64` is exactly:

```
docs/recall-map-field-verification.md
scripts/render_recall_map_relevance_report.py
tests/test_recall_map_relevance_report.py
artifacts/recall-map/relevance/field/protocol.json
```

plus, on later operator commits, the other tracked field artifacts from the
table above. **Any other path rejects the carrier.** The predicates that
actually pin runtime identity — the two `src` git trees, the typed-content
SHA-256, the policy bytes, the selected-policy digest and the five instrument
hashes — are unaffected and remain gating.

## Order of operations

Six steps, strictly in order, per host where noted. Every step is fail-closed:
when a predicate fails, **stop and report the exact protocol conflict**. Never
repair it by refitting the policy, by extending or trimming cohort membership,
by moving a cutoff, or by editing a threshold, floor, instrument or
preregistration.

| # | Step | Host | Gate to proceed |
| --- | --- | --- | --- |
| 1 | Prepare stores | `sfx`, then `alt` | two valid `store-preparation-receipt-v2` |
| 2 | Promote and deploy the frozen candidate | `sfx`, `alt` | `rollout_authorized=true` committed; one `candidate-promotion-receipt-v1` + two `deployment-receipt-v2` |
| 3 | Accrue | `sfx`, `alt` independently | two valid `outcome-blind-accrual-receipt-v1` |
| 4 | Seal the manifest | repository | `manifest.json` committed, outcome-blind |
| 5 | Release the outcome embargo | repository | final `protocol.json` revision with `release_authorized=true` |
| 6 | Run the instruments | repository | the four tracked outputs |

**1. Prepare stores.** Build the `sfx` field store from the byte-bound stable
source, produce the closed `alt` seed from it, hand-carry the seed to `alt`,
build the `alt` field store from those exact bytes. Both stores are proved
against [Fresh-store invariants](#fresh-store-invariants) *before any candidate
process may open them*. Each host writes its own immutable receipt.
*On failure:* stop. Do not search for a substitute source, do not recapture a
snapshot, do not sanitize a drifted database, do not repair inherited rows.

**2. Promote and deploy.** Only after a committed `protocol.json` revision sets
`rollout_authorized=true` and hash-binds both accepted preparation receipts.
Fast-forward the shared `sfx` checkout to the authorized carrier, verify the
carrier contract, then on each host: capture the sanitized before-configuration,
stop the service, bind `rollback-binding-v1` over the now-closed previous
database, switch **only** the database selection to the proven field store,
start exactly once, verify health, a new `boot_id` and served protocol.
*On failure:* stop and roll back per [rollback](#rollback-sfx). Do not restart a
second time to "settle" it, do not edit the unit beyond the bound
database-selection change, do not touch delivery configuration.

**3. Accrue.** Let each host run. Poll structurally only. Choose
`candidate_cutoff_utc` as the *earliest* boundary whose complete prefix meets
every structural floor, then `fixed_as_of_utc = candidate_cutoff_utc + 24h`, and
close on the first poll whose wall clock and closed inputs cover that instant.
*On failure:* if no qualifying cutoff exists yet, keep accruing without
inspecting any outcome. If a floor is met and then a later boundary would look
"better", the earlier boundary still stands — a later cutoff may never replace
the first qualifying one.

**4. Seal the manifest.** From the outcome-blind structural inventory only.
*On failure:* if the historical train/eval identity reference cannot be
reproduced without hash drift, or any required intersection is nonzero, manifest
sealing is forbidden — report it. Never drop a conflicting unit or a host to
manufacture disjointness.

**5. Release the embargo.** A `protocol.json` revision committed strictly after
the manifest commit binds that exact manifest and sets
`outcome_access_embargo.release_authorized=true`. *On failure:* leave the seal
fields absent and `release_authorized=false`.

**6. Run the instruments.** Unchanged, at their frozen hashes, per host, never
pooled. Read the result per [Validity criteria](#validity-criteria).

## Fresh-store invariants

Every one of these must hold, on each host, **before any candidate process opens
the store**. They are proved on *closed* copies — no process holding the file,
opened read-only through the URI `file:<path>?mode=ro&immutable=1`, never
through `MemoryStore`.

1. **Byte-bound stable source.** The only permitted source is
   `/home/sfx/.local/share/living-memory/field/directional-zsum-r1-fresh-field-20260823/sfx/source.sqlite3`:
   a regular file (not a symlink), mode `0444`, 514121728 bytes, sha256
   `9f00abf4467b96bfb6f3cdce9123edfc275a392030da6833e3f7b58b3dcbabcc`, byte-equal
   to the origin observation `/tmp/mapdemo-snap.sqlite3` (same size and hash),
   with provenance evidence `/tmp/mapdemo.py` at 1482 bytes / sha256
   `85956b5fd6b180a4d781ab657fa611b0220696989bdb3b2338daede2278acdb4`.
   `PRAGMA integrity_check` returns `ok`; schema version 3, page size 4096, page
   count 125518, UTF-8; `recall_events` holds 54917 rows spanning
   `2026-05-14T20:05:23Z` … `2026-08-19T18:19:32Z`.
   *If any of these is absent, a symlink, non-regular, wrong-sized or not
   SHA-256 exact: stop. Do not search for a replacement.*

2. **Metadata-only history-isolation transform.** The only permitted logical
   change on the copied field store is
   `UPDATE recall_events SET recall_map = NULL WHERE recall_map IS NOT NULL`
   inside one immediate transaction. The bound source has **zero** such rows, so
   the transform has no target and the prepared store and the `alt` seed must
   stay byte-identical to the source (514121728 / `9f00abf4…`). A nonzero
   pre-transform count is source drift, not permission to sanitize. If the
   `recall_map` column is absent on recheck, that is schema drift — reject; do
   not migrate it during preparation. Forbidden: any row insertion or deletion,
   any change to a `recall_events` column other than `recall_map`, any change to
   `nodes`, `connections`, `query_anchors`, `query_anchor_edges`, retrieval
   policy, feedback or configuration rows, `VACUUM`, dump-and-restore, filtering
   by node identity, or rebuilding through application APIs.

3. **Zero non-NULL `recall_map` rows before deployment.**
   `SELECT count(*) FROM recall_events WHERE recall_map IS NOT NULL` = 0, both
   pre- and post-transform, on the prepared field store, proved before the
   candidate process starts.

4. **Reproduced organic dependency closure.** Over the frozen event interval
   `2026-08-01T00:00:00Z` … `2026-08-19T11:00:00Z` inclusive: every declared
   column of every `recall_events` row in the interval (`ORDER BY id`), the
   complete `query_anchor_edges` table (`ORDER BY anchor_id, target_id`) and the
   complete `query_anchors` table (`ORDER BY id`), preceded by each table's
   `sqlite_schema` SQL and `PRAGMA table_xinfo` metadata. Schema digest,
   per-table row counts, per-table digests and the combined digest must be
   byte-equal between the closed source and the prepared field store;
   `organic_closure_equal = true`. Separately, the sealed organic aggregate must
   reproduce exactly: the 1154-byte canonical `arms.organic` object hashes to
   `cc2f1a645d8031c22552f24417d3b1411ec6f9ccad124b28f0fadb23c70f5fac`, with
   primary items 4511 / consumed 2103 / rate 0.4662, anchor items 3703 /
   consumed 1500 / rate 0.4051, 5189 events in window, 1740 events scored, 8214
   items, 1055 transport sessions.

5. **Inherited foreign-key baseline** (`sqlite-foreign-key-inherited-baseline-v1`).
   The source carries 35 pre-existing violation tuples, all in `recall_events`,
   with 0 in the sealed organic closure. These are *inherited*: never repair,
   remove or renumber a row. Required:
   * source metadata reproduces 18 user tables, 8 foreign-key declarations, and
     the fully specified supplementary digest — 2314 bytes,
     `c32fa5bfb5c8d5fc74bb9c36b708ce2d3eb1bdc111625d6a4bd75c2b0540c8eb`;
   * source↔field equality of `metadata_user_table_count`,
     `metadata_declaration_count`, `metadata_serialized_bytes`, `metadata_sha256`,
     `tuple_cursor_metadata_sha256`, `tuple_count`, `tuple_serialized_bytes` and
     `tuples_sha256` — all four equality booleans true;
   * `new_violations = 0` by multiset subtraction (field minus source);
   * `source_closure_violations = field_closure_violations = closure_violations = 0`.

   A `foreign_key_check` tuple naming a closure table with a NULL or
   unresolvable `rowid` is a protocol conflict, not an "outside the closure"
   assumption.

6. **Zero-WAL closure.** Before every copy: no process holds the main file, and
   the `-wal` sidecar is absent or exactly 0 bytes. A nonzero WAL is an immediate
   protocol conflict — do not checkpoint it. A stale `-shm` (32768 bytes is the
   observed state) may be ignored only after proving no open handle and an
   absent-or-zero WAL. Hash the main file before and after any read performed by
   an unchanged tool.

7. **Non-aliased, pairwise-distinct roots.** `${HOME}` is resolved on the named
   host before any copy; the receipt records the absolute realpath. No root may
   be a symlink, alias a live store, or resolve inside another bound root.

   | Role | `sfx` | `alt` |
   | --- | --- | --- |
   | Field store root | `/home/sfx/.local/share/living-memory/field/directional-zsum-r1-fresh-field-20260823/sfx` | `${HOME}/.local/share/living-memory/field/directional-zsum-r1-fresh-field-20260823/alt` |
   | Source snapshot | `…/sfx/source.sqlite3` | — |
   | Field store | `…/sfx/field.sqlite3` | `…/alt/field.sqlite3` |
   | Seed | `…/sfx/alt-seed.sqlite3` | `…/alt/received-seed.sqlite3` |
   | AE journal archive root | `/home/sfx/.local/share/ae/field/directional-zsum-r1-fresh-field-20260823/sfx` | `${HOME}/.local/share/ae/field/directional-zsum-r1-fresh-field-20260823/alt` |
   | Rollback material root | `/home/sfx/.local/share/living-memory/rollback/directional-zsum-r1-fresh-field-20260823/sfx` | `${HOME}/.local/share/living-memory/rollback/directional-zsum-r1-fresh-field-20260823/alt` |

   Within a host, field-store, AE-journal and rollback roots are different,
   non-nested directories. Across hosts, roots are distinct by the tuple
   `(host_id, filesystem identity, absolute realpath)` — identical path *text* on
   two hosts never merges evidence. Neither field store may resolve to the
   authoritative stable source, to the mutable live store, or to the other host's
   store.

### The structural check helper

Invariants 3–5 need one small program. Save it **outside the repository** as
`$NSROOT/tools/fieldcheck.py` on each host, where
`NSROOT=~/.local/share/living-memory/field/directional-zsum-r1-fresh-field-20260823`.
It reads only schema metadata, PRAGMA output, counts and digests, and never
prints a row value.

The typed length-prefix encoding pins NULL (tag `N`, zero length) and the payload
form of each storage class, but not the literal tag bytes for the other classes.
Choose the assignment below **once** and use the identical program for the source
copy and the field copy on both hosts: every predicate over these digests is a
source↔field equality, plus counts and byte lengths that no tag choice can move.
The one digest compared against a preseal constant — the supplementary
foreign-key digest `c32fa5bf…` — is fully specified as JSON and is independent of
the tag assignment.

```python
#!/usr/bin/env python3
"""Outcome-blind structural checks over a CLOSED SQLite copy. Metadata only."""
import hashlib, json, sqlite3, struct, sys

CLOSURE_START, CLOSURE_END = "2026-08-01T00:00:00Z", "2026-08-19T11:00:00Z"
CLOSURE_TABLES = ("recall_events", "query_anchor_edges", "query_anchors")
ORDER = {"recall_events": "id", "query_anchor_edges": "anchor_id, target_id",
         "query_anchors": "id"}

def enc(tag, payload):            # one-byte tag || u64be length || payload
    return tag + struct.pack(">Q", len(payload)) + payload

def val(v):
    if v is None:                 return enc(b"N", b"")
    if isinstance(v, bool):       v = int(v)
    if isinstance(v, int):        return enc(b"I", str(v).encode("ascii"))
    if isinstance(v, float):      return enc(b"R", struct.pack(">d", v))
    if isinstance(v, str):        return enc(b"T", v.encode("utf-8"))
    if isinstance(v, (bytes, bytearray)): return enc(b"B", bytes(v))
    raise SystemExit("unknown storage class")

def q(name):  return '"' + name.replace('"', '""') + '"'
def sha(b):   return hashlib.sha256(b).hexdigest()
def jc(o):    return json.dumps(o, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")).encode("utf-8")

def ro(path):
    return sqlite3.connect("file:%s?mode=ro&immutable=1" % path, uri=True)

def fk_metadata(con):
    names = sorted((r[0] for r in con.execute(
        "SELECT name FROM sqlite_schema WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%'")), key=lambda n: n.encode("utf-8"))
    stream, decls = b"", 0
    for name in names:
        sql = con.execute("SELECT sql FROM sqlite_schema WHERE type='table' "
                          "AND name=?", (name,)).fetchone()[0]
        cur = con.execute("PRAGMA foreign_key_list(%s)" % q(name))
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        decls += len(rows)
        recs = sorted(b"".join(val(v) for v in row) for row in rows)
        stream += enc(b"H", name.encode("utf-8")) + val(sql)
        stream += enc(b"C", jc(cols)) + enc(b"L", struct.pack(">Q", len(recs)))
        for rec in recs:
            stream += enc(b"F", rec)
    return {"metadata_user_table_count": len(names),
            "metadata_declaration_count": decls,
            "metadata_serialized_bytes": len(stream),
            "metadata_sha256": sha(stream)}

def fk_violations(con):
    cur = con.execute("PRAGMA foreign_key_check")
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchall()                       # fetched to exhaustion, never capped
    meta = jc(cols)
    recs = sorted(b"".join(val(v) for v in row) for row in rows)
    stream = b"lm-field-fk-tuples-v1\0" + meta + struct.pack(">Q", len(recs))
    for rec in recs:
        stream += struct.pack(">Q", len(rec)) + rec
    return ({"tuple_cursor_metadata_sha256": sha(b"lm-field-fk-cursor-v1\0" + meta),
             "tuple_count": len(recs), "tuple_serialized_bytes": len(stream),
             "tuples_sha256": sha(stream)}, cols, rows, recs)

def fk_supplementary(con):
    cur = con.execute("PRAGMA foreign_key_check")
    cols = [d[0] for d in cur.description] if cur.description else []
    encoded = sorted((jc(dict(zip(cols, row))) for row in cur.fetchall()))
    payload = jc({"columns": cols, "rows": [json.loads(e) for e in encoded]})
    return {"supplementary_canonical_json_bytes": len(payload),
            "supplementary_canonical_json_sha256": sha(payload)}

def closure_rowids(con):
    ids = {"recall_events": set(r[0] for r in con.execute(
               "SELECT rowid FROM recall_events WHERE created_at >= ? "
               "AND created_at <= ?", (CLOSURE_START, CLOSURE_END)))}
    for t in ("query_anchor_edges", "query_anchors"):
        ids[t] = set(r[0] for r in con.execute("SELECT rowid FROM %s" % q(t)))
    return ids

def closure_violations(con, cols, rows):
    ids, n = closure_rowids(con), 0
    ix = {c: i for i, c in enumerate(cols)}
    for row in rows:
        table, rowid = row[ix["table"]], row[ix["rowid"]]
        if table in ids:
            if rowid is None or not isinstance(rowid, int):
                raise SystemExit("protocol conflict: closure tuple with no rowid")
            if rowid in ids[table]:
                n += 1
    return n

def closure_digest(con):
    schema = b""
    for t in CLOSURE_TABLES:
        sql = con.execute("SELECT sql FROM sqlite_schema WHERE type='table' "
                          "AND name=?", (t,)).fetchone()[0]
        cur = con.execute("PRAGMA table_xinfo(%s)" % q(t))
        schema += enc(b"H", t.encode("utf-8")) + val(sql)
        schema += enc(b"C", jc([d[0] for d in cur.description]))
        for row in cur.fetchall():
            schema += b"".join(val(v) for v in row)
    per, combined, counts = {}, schema, {}
    for t in CLOSURE_TABLES:
        where = ("WHERE created_at >= '%s' AND created_at <= '%s' "
                 % (CLOSURE_START, CLOSURE_END)) if t == "recall_events" else ""
        cur = con.execute("SELECT * FROM %s %sORDER BY %s" % (q(t), where, ORDER[t]))
        body, n = enc(b"H", t.encode("utf-8")), 0
        for row in cur:
            body += b"".join(val(v) for v in row)
            n += 1
        body = enc(b"L", struct.pack(">Q", n)) + body
        per[t], counts[t], combined = sha(body), n, combined + body
    return {"closure_schema_sha256": sha(schema), "closure_table_sha256": per,
            "closure_table_counts": counts, "closure_combined_sha256": sha(combined)}

def report(path):
    con = ro(path)
    out = {"path": path,
           "sqlite_integrity_ok": con.execute("PRAGMA integrity_check").fetchone()[0] == "ok",
           "nonnull_recall_map_rows": con.execute(
               "SELECT count(*) FROM recall_events WHERE recall_map IS NOT NULL").fetchone()[0]}
    out.update(fk_metadata(con))
    tup, cols, rows, recs = fk_violations(con)
    out.update(tup)
    out.update(fk_supplementary(con))
    out["closure_violation_count"] = closure_violations(con, cols, rows)
    out.update(closure_digest(con))
    out["_records"] = [rec.hex() for rec in recs]      # for multiset subtraction only
    con.close()
    return out

if __name__ == "__main__":
    src, fld = report(sys.argv[1]), report(sys.argv[2])
    a, b = sorted(src.pop("_records")), sorted(fld.pop("_records"))
    from collections import Counter
    new = sum((Counter(b) - Counter(a)).values())
    print(json.dumps({"source": src, "field": fld, "new_violations": new}, indent=1))
```

Accept only when, for the pair `(source, field)`:
`sqlite_integrity_ok` true on both; `nonnull_recall_map_rows` 0 on both;
source `metadata_user_table_count` 18, `metadata_declaration_count` 8,
`tuple_count` 35, `supplementary_canonical_json_bytes` 2314 and
`supplementary_canonical_json_sha256` `c32fa5bf…`; all eight metadata/tuple
fields equal source↔field; every `closure_*` field equal source↔field;
`new_violations` 0; and `closure_violation_count` 0 on both.

## Host commands: sfx

Run these on `sfx`, as the operator, in this order. Set the environment first:

```sh
NS=directional-zsum-r1-fresh-field-20260823
NSROOT=/home/sfx/.local/share/living-memory/field/$NS
FIELD=$NSROOT/sfx
RECEIPTS=$NSROOT/receipts
ROLLBACK=/home/sfx/.local/share/living-memory/rollback/$NS/sfx
AEARCHIVE=/home/sfx/.local/share/ae/field/$NS/sfx
CHECKOUT=/home/sfx/p/lm
mkdir -p "$FIELD" "$RECEIPTS" "$ROLLBACK" "$AEARCHIVE" "$NSROOT/tools" \
         "$NSROOT/operator-inputs/alt"
```

### Store preparation (sfx)

```sh
# 1. the bound source, unchanged and closed
stat -c '%F %a %s %n' "$FIELD/source.sqlite3"   # regular file 444 514121728
sha256sum "$FIELD/source.sqlite3"               # 9f00abf4…
stat -c '%s' /tmp/mapdemo-snap.sqlite3; sha256sum /tmp/mapdemo-snap.sqlite3
cmp "$FIELD/source.sqlite3" /tmp/mapdemo-snap.sqlite3 && echo byte-equal
stat -c '%s' /tmp/mapdemo.py; sha256sum /tmp/mapdemo.py   # 1482 / 85956b5f…
ls -l "$FIELD/source.sqlite3-wal" 2>/dev/null   # absent, or exactly 0 bytes
fuser "$FIELD/source.sqlite3" >/dev/null 2>&1 && echo 'OPEN HANDLE — STOP'
#   (no output from fuser == no open handle; `lsof -t -- <path>` is equivalent)
```

```sh
# 2. closed byte-copy -> field store (fsync + atomic rename, never SQLite backup)
closedcopy() {   # closedcopy <src> <dst> ; both must be closed regular files
  python3 - "$1" "$2" <<'PY'
import hashlib, os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
assert not os.path.exists(dst), "destination already present — verify or move aside"
assert os.path.isfile(src) and not os.path.islink(src), "source is not a regular file"
def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
before = digest(src)
tmp = dst + ".tmp"
shutil.copyfile(src, tmp)
with open(tmp, "rb+") as fh:
    os.fsync(fh.fileno())
os.rename(tmp, dst)
d = os.open(os.path.dirname(dst), os.O_DIRECTORY); os.fsync(d); os.close(d)
assert before == digest(dst) == digest(src), "hash drift during copy — STOP"
print(before, os.path.getsize(dst), dst)
PY
}
closedcopy "$FIELD/source.sqlite3" "$FIELD/field.sqlite3"
cmp "$FIELD/source.sqlite3" "$FIELD/field.sqlite3" && echo byte-equal
```

```sh
# 3. structural proof on both CLOSED copies
python3 "$NSROOT/tools/fieldcheck.py" "$FIELD/source.sqlite3" "$FIELD/field.sqlite3" \
    > "$NSROOT/tools/sfx-structural.json"
```

The transform has no target on this source (`nonnull_recall_map_rows` is 0 both
pre and post), so record `pre_transform_nonnull_recall_map_rows = 0`,
`post_transform_nonnull_recall_map_rows = 0` and the SHA-256 of the *unexecuted*
statement text `UPDATE recall_events SET recall_map = NULL WHERE recall_map IS NOT NULL`
as `transform_statement_sha256`. If the count is nonzero, that is source drift —
stop, do not sanitize.

```sh
# 4. sealed-instrument check + the two allowlisted organic verifications
cd "$CHECKOUT"
for f in artifacts/recall-map/prereg.json scripts/recall_map_effect.py \
         scripts/recall_map_latency_bench.py; do
  printf '%s %s %s\n' "$(stat -c%s $f)" "$(sha256sum $f | cut -d' ' -f1)" "$f"; done
for f in /home/sfx/p/ae/docs/agent-stream-prereg.md \
         /home/sfx/p/ae/tools/lm_workload/stream_injection_metrics.py; do
  printf '%s %s %s\n' "$(stat -c%s $f)" "$(sha256sum $f | cut -d' ' -f1)" "$f"; done

PYTHONPATH=src python3 scripts/recall_map_effect.py --verify-prereg \
    --as-of 2026-08-19T11:00:00Z
sha256sum "$FIELD/field.sqlite3"                       # before the tool reads it
PYTHONPATH=src python3 scripts/recall_map_effect.py --db "$FIELD/field.sqlite3" \
    --as-of 2026-08-19T11:00:00Z --no-positional --quiet --out /tmp/organic-sfx.json
sha256sum "$FIELD/field.sqlite3"                       # unchanged after
python3 - <<'PY'
import hashlib, json
arms = json.load(open("/tmp/organic-sfx.json"))["arms"]["organic"]
b = json.dumps(arms, ensure_ascii=False, sort_keys=True,
               separators=(",", ":")).encode("utf-8")
print(len(b), hashlib.sha256(b).hexdigest())   # 1154 cc2f1a64…
PY
```

Expected against the sealed constants: 1154 bytes,
`cc2f1a645d8031c22552f24417d3b1411ec6f9ccad124b28f0fadb23c70f5fac`, primary
4511/2103/0.4662, anchor 3703/1500/0.4051. Any mismatch is a protocol conflict —
do not amend the evaluator or the preregistration.

```sh
# 5. the closed alt seed — copied from the proven field store, never reopened
closedcopy "$FIELD/field.sqlite3" "$FIELD/alt-seed.sqlite3"
cmp "$FIELD/field.sqlite3" "$FIELD/alt-seed.sqlite3" && echo byte-equal
sha256sum "$FIELD/alt-seed.sqlite3"    # 9f00abf4… / 514121728 bytes
chmod 0444 "$FIELD/alt-seed.sqlite3"
```

```sh
# 6. the immutable receipt (schema: store-preparation-receipt-v2)
#    Assemble the JSON by hand from the recorded values — every required field
#    from "Receipt schemas", nothing else — then publish it atomically:
publish() {   # publish <draft-json> <destination>
  python3 - "$1" "$2" <<'PY'
import hashlib, json, os, sys
draft, dst = sys.argv[1], sys.argv[2]
raw = open(draft, "rb").read()
json.loads(raw)                       # must parse; unknown fields reject at binding
assert not os.path.exists(dst), "receipt already published — never rewrite one"
tmp = os.path.join(os.path.dirname(dst), "." + os.path.basename(dst) + ".tmp")
fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.write(fd, raw); os.fsync(fd); os.close(fd)
os.chmod(tmp, 0o444)
os.rename(tmp, dst)
d = os.open(os.path.dirname(dst), os.O_DIRECTORY); os.fsync(d); os.close(d)
print(len(raw), hashlib.sha256(raw).hexdigest(), dst)
PY
}
publish sfx-store-preparation.draft.json "$RECEIPTS/sfx-store-preparation.json"
ls -l "$RECEIPTS/sfx-store-preparation.json"     # mode 0444, regular file
```

`publish` is reused for every receipt on both hosts; its printed byte length and
SHA-256 are the values bound in the next `protocol.json` or `manifest.json`
revision. Delete the draft only after the published bytes verify.

Then commit a `protocol.json` revision that binds both accepted preparation
receipts and sets `rollout_authorized=true`. **Do not proceed until that
revision is committed.**

### Candidate promotion (sfx)

The shared checkout `/home/sfx/p/lm` is promoted **by the operator only**, by
fast-forward, with the service *not* restarted in this step.

```sh
cd "$CHECKOUT"
git status --porcelain            # must be empty: worktree_clean_before
BEFORE=$(git rev-parse HEAD)
git pull --ff-only
CARRIER=$(git rev-parse HEAD)
git status --porcelain            # must be empty: worktree_clean_after

# carrier contract
git merge-base --is-ancestor 5b7f4c0c3ba94b1d4d36fc0b9fcb00a6e8bcdf64 $CARRIER && echo ok
git merge-base --is-ancestor 24db9bbb33f30e507a1b58af7eedc6163bc962ab $CARRIER && echo ok
git diff --name-only 5b7f4c0c3ba94b1d4d36fc0b9fcb00a6e8bcdf64 $CARRIER
#   -> exactly the four paths listed in "Scope and execution boundary",
#      plus later tracked field artifacts. Anything else REJECTS.

git rev-parse $CARRIER:src            # f45eeb5a17375340c6f3a7dc6ec7674f96b251df
git rev-parse $CARRIER:src/living_memory  # 0181c33c6d3d0763beb856176a9f0cdb155a5b6c
git rev-parse $CARRIER^{tree}         # record as carrier_tree

# the two allowlisted non-runtime blobs: mode, bytes, blob, sha256 — all exact
git ls-tree $CARRIER scripts/render_recall_map_relevance_report.py \
                     tests/test_recall_map_relevance_report.py
stat -c '%s' scripts/render_recall_map_relevance_report.py   # 43205
sha256sum scripts/render_recall_map_relevance_report.py      # 4f8f0c07…
stat -c '%s' tests/test_recall_map_relevance_report.py       # 6768
sha256sum tests/test_recall_map_relevance_report.py          # 277ab006…

# deployed-package typed content -> b302ede4…
python3 - <<'PY'
import hashlib, subprocess, struct
out = subprocess.run(["git", "ls-tree", "-r", "-z", "--full-tree", "HEAD",
                      "--", "src/living_memory/"], capture_output=True, check=True).stdout
entries = []
for rec in out.split(b"\0"):
    if not rec:
        continue
    meta, path = rec.split(b"\t", 1)
    mode, kind, blob = meta.split()
    if kind != b"blob":
        continue
    entries.append((path, mode, blob))
entries.sort(key=lambda e: e[0])
h = hashlib.sha256()
def field(tag, payload):
    h.update(tag + struct.pack(">Q", len(payload)) + payload)
for path, mode, blob in entries:
    data = subprocess.run(["git", "cat-file", "blob", blob.decode()],
                          capture_output=True, check=True).stdout
    field(b"P", path); field(b"M", mode); field(b"B", data)
print(h.hexdigest(), len(entries))     # b302ede4…  49
PY

# policy + runtime constant
stat -c '%s' artifacts/recall-map/relevance/policy.json      # 16815
sha256sum artifacts/recall-map/relevance/policy.json         # 1f065b9e…
python3 - <<'PY'
import hashlib, json, sys
sys.path.insert(0, "src")
from living_memory.recall_map import RELEVANCE_POLICY_DIGEST
sel = json.load(open("artifacts/recall-map/relevance/policy.json"))["selected_policy"]
b = json.dumps(sel, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
d = hashlib.sha256(b).hexdigest()
print(len(b), d, RELEVANCE_POLICY_DIGEST, d == RELEVANCE_POLICY_DIGEST)
#   7750  3acad3d9…  3acad3d9…  True
PY
```

Publish `sfx-candidate-promotion.json` (`candidate-promotion-receipt-v1`) with
`service_restarted=false` and `outcome_accessed=false`.

### Deployment (sfx)

The unit is the user service documented in `docs/deployment.md`:
`living-memory.service`, with `living-memory-tls.service` `PartOf=` it. The
**only** permitted configuration change is the database selection — every
delivery trigger, renderer, chat path, cap, silence rule, probe, endpoint, scope
and wire budget stays byte-for-byte identical, and the AE delivery configuration
is not touched at all.

```sh
# 1. sanitized before-configuration (host-configuration-v1) and pre-restart boot id
systemctl --user cat living-memory.service living-memory-tls.service
systemctl --user show -p FragmentPath -p ExecStart --value living-memory.service
sha256sum ~/.config/systemd/user/living-memory.service
curl -fsS http://127.0.0.1:8765/health          # record boot_id -> before_boot_id
curl -fsS http://127.0.0.1:8765/admin/info
python3 -m pip show living-memory | grep -i 'editable project location'
env | grep -E '^(LM_|LIVING_MEMORY_|AE_STREAM_|AE_CHAT_|STATE_DIR|PROJECT_NAME|LM_URL)='
```

Record every variable named in `schemas.host_configuration.lm_environment_names`
and every `delivery_channel` field, with unset recorded explicitly. Never record
a value matching `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL` or a private key —
presence and source only.

```sh
# 2. stop the service, then bind rollback material over the now-CLOSED live store
systemctl --user stop living-memory.service
systemctl --user is-active living-memory.service living-memory-tls.service  # inactive
LIVE=/home/sfx/.local/share/living-memory/global.sqlite3
LIVEDIR=$(dirname "$LIVE"); LIVEBASE=$(basename "$LIVE")
fuser "$LIVE" >/dev/null 2>&1 && echo 'STILL OPEN — STOP'
( cd "$LIVEDIR" && ls -1 "$LIVEBASE" "$LIVEBASE-wal" "$LIVEBASE-shm" 2>/dev/null ) \
    > "$ROLLBACK/predeployment-store.parts"   # -> predeployment_store_parts
tar -C "$LIVEDIR" -cf "$ROLLBACK/predeployment-store.tar" \
    -T "$ROLLBACK/predeployment-store.parts"
chmod 0444 "$ROLLBACK/predeployment-store.tar"
sha256sum "$LIVE" "$ROLLBACK/predeployment-store.tar"
stat -c '%s' "$LIVE" "$ROLLBACK/predeployment-store.tar"
```

Write `rollback-binding-v1` now, with `predeployment_store_closed=true` and
`restore_material_available=true`. **Do not start the candidate until
`rollback_ready=true`.**

```sh
# 3. change ONLY the database selection, via a drop-in
mkdir -p ~/.config/systemd/user/living-memory.service.d
cat > ~/.config/systemd/user/living-memory.service.d/field-store.conf <<EOF
[Service]
ExecStart=
ExecStart=%h/.local/bin/living-memory-server --transport http --host 127.0.0.1 \
 --port 8765 --db $FIELD/field.sqlite3 --default-scope global
EOF
systemctl --user daemon-reload
systemctl --user show -p ExecStart --value living-memory.service   # only --db differs
sha256sum "$FIELD/field.sqlite3"      # initial_store_sha256, still 9f00abf4…
```

The `ExecStart=` reset line plus one replacement is what keeps every other
argument identical; diff the shown `ExecStart` against the captured
before-value and confirm the *only* difference is the `--db` path.

```sh
# 4. start exactly once, and prove the process is the candidate
systemctl --user start living-memory.service
systemctl --user is-active living-memory.service living-memory-tls.service
curl -fsS http://127.0.0.1:8765/health     # new boot_id != before_boot_id
curl -fsS http://127.0.0.1:8765/admin/info # started_at -> candidate_deployment_instant_utc,
                                           # process_id, argv, default_scope
systemctl --user show -p NRestarts --value living-memory.service   # 0 restarts since start
cd "$CHECKOUT" && python3 scripts/check_deployed_protocol.py --json \
    > "$NSROOT/tools/sfx-served.json"; echo "exit=$?"      # must be 0
sha256sum "$NSROOT/tools/sfx-served.json"
```

`restart_count` in the receipt is 1 — the single start of this deployment. A
`boot_id` equal to `before_boot_id` means the restart did not happen: stop.
`/health` answers `503` without `"ok"` while a start is in flight — poll, do not
restart again.

```sh
# 5. cache generation id (must be fresh, and differ from alt and every prior process)
python3 - <<'PY'
import hashlib, json
obj = {"protocol_id": "directional-zsum-r1-fresh-field-20260823", "host_id": "sfx",
       "after_boot_id": "…", "process_started_at_utc": "…",
       "field_store_initial_sha256": "9f00abf4467b96bfb6f3cdce9123edfc275a392030da6833e3f7b58b3dcbabcc",
       "carrier_commit": "…",
       "source_subtree_typed_content_sha256": "b302ede43343532dddda437c12c8b3ed7a54d07a18b9c67248dfe9944b1eb4ce",
       "selected_policy_sha256": "3acad3d92db2538bf3096ab99d2c4d337ea3c8aebddc226646527b4d277660ea"}
print(hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")).encode("utf-8")).hexdigest())
PY
```

Publish `sfx-deployment.json` (`deployment-receipt-v2`), then re-verify the
before/after delivery-channel digests are equal and `unchanged=true`.

### Accrual (sfx)

Poll structurally only. Never open a `consume`, `curtail`, `probe` or
`capability` record, never read `query`, `results` or any `ambient_context`
member except `task_pattern`.

The permitted recall-event projection is:

```sql
SELECT id, created_at, transport_session_id, session_id, scope, task,
       json_extract(ambient_context, '$.task_pattern') AS task_pattern,
       recall_map
  FROM recall_events
 WHERE recall_map IS NOT NULL
   AND created_at > :candidate_deployment_instant_utc
 ORDER BY created_at, id;
```

`recall_map` may be used only to test NULL vs non-NULL, validate the additive
payload shape, count opportunity-bearing events and ordered offered items, and
derive item positions and exposed-item identities. `scope`, `task` and
`task_pattern` may be passed only through the frozen `cache_key` and never
emitted or grouped.

The permitted AE journal projection is `inject` and `session` records, fields
`v`, `kind`, `session_id`, `seq`, `delivered`, `created_at`/`timestamp`, and
`position.tool_calls_seen`. An injection is *structurally scorable* when its
session shows at least five subsequent tool calls after the inject position.
Any field or record kind not on that allowlist: fail closed before reading it.

**The stopping rule.** Per host, never pooled:

1. Sort the unique timestamps of all structurally eligible post-deployment map
   events and delivered injections. Equal timestamps form one indivisible
   boundary.
2. For each candidate cutoff `C` in ascending order, take the complete prefix
   `timestamp <= C`. Use allowlisted `session` records through `C + 24h` only to
   decide whether injections in that prefix are structurally scorable.
3. Choose the **earliest** `C` whose prefix has `map_events >= 20`,
   `map_items >= 20`, distinct map transport sessions `>= 10`,
   `injections_scorable >= 20` and `injections_scorable / injections_total >= 0.50`.
   No later cutoff may replace it.
4. `fixed_as_of_utc = C + 24h`. Close on the first structural poll whose wall
   clock and closed inputs cover that instant. Records after `fixed_as_of_utc`
   may prove archive closure but may not change membership or floors.
5. If no qualifying `C` exists, keep accruing without inspecting any outcome. On
   a protocol conflict, unknown field, missing timestamp or embargo violation:
   fail closed. Do not skip the unit, move the cutoff, or substitute the other
   host.

Identity digests, computed with the frozen implementation from the authorized
checkout (`source_identity` is the literal string `local` on both hosts):

| Domain | Formula |
| --- | --- |
| event | `sha256("recall-map-relevance-event-v1\0" + source_identity + "\0" + event_id)` |
| transport session | `sha256("lm-field-transport-v1\0" + source_identity + "\0transport\0" + transport_session_id)` |
| session | `sha256("lm-field-session-v1\0" + source_identity + "\0session\0" + session_id)` |
| cache | `sha256("lm-field-cache-v1\0" + source_identity + "\0cache\0" + cache_key(scope or "global", task=task, task_pattern=task_pattern))` |

```python
import sys; sys.path.insert(0, "src")
from living_memory.recall_map import cache_key      # the frozen implementation
```

Per host and domain, parse each lower-case hex identity to 32 raw bytes,
deduplicate, sort lexicographically, and emit only `(domain, count, digest)`
where the digest is
`sha256(b"lm-field-identity-set-v1\0" + domain.encode() + b"\0" + u64be(count) + concat(values))`.
Deduplicate only after domain separation. Event identity and cache identity are
mandatory for every admitted map event; a missing `transport_session_id` or
`session_id` contributes no identity in that domain and may never be invented.
Host id is recorded separately and is never mixed into a key.

```sh
# close and hash the inputs, then publish the accrual receipt
sha256sum "$FIELD/field.sqlite3"        # after closing: closed_store_snapshot_sha256
JDIR='<AE_STREAM_JOURNAL_DIR as recorded in host_configuration.delivery_channel>'
tar -C "$(dirname "$JDIR")" -cf "$AEARCHIVE/sfx-journals.tar" "$(basename "$JDIR")"
chmod 0444 "$AEARCHIVE/sfx-journals.tar"; sha256sum "$AEARCHIVE/sfx-journals.tar"
```

The AE journals keep being written to their unchanged producer roots; the
archive is a blind copy made **after** accrual closes, without inspecting record
contents. Publish `sfx-accrual.json` (`outcome-blind-accrual-receipt-v1`).

### Rollback (sfx)

Triggered by any of: carrier / source / policy / instrument / store / root /
cache-generation / served-provenance / receipt mismatch; health or
served-protocol failure; delivery-channel, payload, budget, silence, cap or
persisted-filter-accounting drift; an embargo violation or any attempt to score
before release; an operator-directed abort.

```sh
systemctl --user stop living-memory.service
rm ~/.config/systemd/user/living-memory.service.d/field-store.conf
rmdir ~/.config/systemd/user/living-memory.service.d 2>/dev/null
systemctl --user daemon-reload
# restore the exact closed bundle bound by rollback-binding-v1
tar -C "$(dirname $LIVE)" -xf "$ROLLBACK/predeployment-store.tar"
sha256sum "$LIVE"                       # equals predeployment_db_sha256
systemctl --user start living-memory.service            # exactly once
curl -fsS http://127.0.0.1:8765/health                  # new boot id, ok
cd "$CHECKOUT" && python3 scripts/check_deployed_protocol.py; echo "exit=$?"  # 0
```

Restore the previous carrier, install mode, distribution location, database
path, service argv, sanitized configuration and delivery configuration exactly
as bound. Then: preserve the candidate field store, seed, snapshot, journal
archive and receipts **read-only** for audit — never delete, rewrite, reopen
through application APIs, merge or reuse them. Mark that host's cohort void and
publish `sfx-rollback.json` (`rollback-receipt-v1`). A void cohort needs a new
protocol id, a fresh store, a fresh cache generation and new receipts; it is
never re-scored. Rollback on one host neither supplies evidence for nor changes
membership on the other, and never sets `rollout_authorized` or
`release_authorized` to true.

## Host commands: alt

`alt` is reached and operated **only by the operator, from `alt` itself**. No
agent connects to it; no `ssh`, `scp`, `rsync`, remote shell, remote copy,
package install, service restart, health check or receipt command is run against
it from anywhere else. Neither `alt` receipt may be generated or edited off-host
and labelled as on-host evidence.

Every step of the `sfx` sequence applies unchanged in substance; the differences
are listed below. Resolve `${HOME}` on `alt` and record absolute realpaths:

```sh
NS=directional-zsum-r1-fresh-field-20260823
NSROOT=$HOME/.local/share/living-memory/field/$NS
FIELD=$NSROOT/alt
RECEIPTS=$FIELD/receipts
ROLLBACK=$HOME/.local/share/living-memory/rollback/$NS/alt
AEARCHIVE=$HOME/.local/share/ae/field/$NS/alt
ALT_CHECKOUT='<the operator-owned Living Memory checkout on alt>'
mkdir -p "$FIELD" "$RECEIPTS" "$ROLLBACK" "$AEARCHIVE" "$NSROOT/tools"
readlink -f "$FIELD" "$AEARCHIVE" "$ROLLBACK"   # non-symlink, non-nested, distinct
```

`fieldcheck.py`, `closedcopy` and `publish` are the same three helpers as on
`sfx` — copy them to `alt` and define them there before the steps below.
`fieldcheck.py` in particular must be **byte-identical** to the one used on
`sfx`, so the tag assignment is shared.

### Store preparation (alt)

1. Do not copy anything until `protocol.json` carries the sfx-bound expected seed
   byte length and SHA-256. `rollout_authorized` is still `false` at this stage.
2. Hand-carry `alt-seed.sqlite3` to `alt` and perform every filesystem copy from
   an interactive process **on** `alt`:

```sh
# after the manual transfer to $FIELD/received-seed.sqlite3
stat -c '%s' "$FIELD/received-seed.sqlite3"      # 514121728
sha256sum "$FIELD/received-seed.sqlite3"         # 9f00abf4…
chmod 0444 "$FIELD/received-seed.sqlite3"
closedcopy "$FIELD/received-seed.sqlite3" "$FIELD/field.sqlite3"   # same helper as sfx
cmp "$FIELD/received-seed.sqlite3" "$FIELD/field.sqlite3" && echo byte-equal
```

3. Verify size and SHA-256 **before any application opens the bytes**, then run
   the identical proofs as on `sfx` — `fieldcheck.py` over
   `(received-seed.sqlite3, field.sqlite3)`, the allowlisted organic
   reproduction, path distinctness, policy and all five instrument hashes —
   without printing row values or touching any candidate outcome.
4. Publish `alt-store-preparation.json` on `alt` at
   `$RECEIPTS/alt-store-preparation.json` using `store-preparation-receipt-v2`,
   atomic rename, mode `0444`. Then transfer **only those immutable bytes** to
   `/home/sfx/.local/share/living-memory/field/$NS/operator-inputs/alt/alt-store-preparation.json`
   and re-verify length and SHA-256 there. Do not deploy or restart yet.

### Deployment (alt)

Preconditions: a committed `protocol.json` revision sets
`rollout_authorized=true` and hash-binds both accepted preparation receipts and
exact initial store bytes; the authorized carrier and all bound LM and AE
instrument bytes are available on `alt`; the embargo is still active and no
candidate outcome has been inspected.

1. Capture the pre-change host configuration and the pre-restart `/health`
   `boot_id` on `alt`, including the complete sanitized LM and AE delivery
   configuration required by `host-configuration-v1`.
2. **Commit delivery.** `alt` has no non-interactive GitHub access. Deliver the
   carrier as an operator-supplied bundle and fast-forward locally, then verify
   the carrier contract exactly as on `sfx` before installing anything:

```sh
git -C "$ALT_CHECKOUT" pull --ff-only /tmp/'<name>'.bundle '<branch>'
```

   (an operator-owned temporary local branch is the alternative).
3. **Install mode.** Inspect `pip show living-memory` on `alt`. If it reports an
   `Editable project location`, confirm it is the authorized checkout and
   reinstall only if dependencies or project scripts changed. If it is a copied
   package, run that environment's
   `pip install --no-deps --force-reinstall <authorized-checkout>` before the
   restart. Reinstalling under a live daemon is safe: the running process holds
   its modules in memory.
4. Change **only** the Living Memory database selection to the proven
   `$FIELD/field.sqlite3`. Preserve every delivery trigger, renderer, chat, cap,
   silence, probe, endpoint, scope and wire-budget value byte-for-byte; the AE
   delivery configuration is untouched.
5. Stop the actual `alt` `living-memory` unit using **that host's** existing unit
   scope (`systemctl --user …` for a user unit, `sudo systemctl …` for a system
   unit), bind the now-closed prior main file and every present sidecar into
   `rollback-binding-v1`, then apply the database-selection change and start the
   service exactly once. Do not start until `rollback_ready=true`.
6. From `alt` itself: both relevant units active, read localhost `/health` and
   `/admin/info`, require a new `boot_id`, and record the actual UTC
   `started_at`, PID, effective argv, default scope and candidate deployment
   instant.
7. From the authorized checkout **on `alt`**, run
   `python3 scripts/check_deployed_protocol.py --default-scope <alt's scope>` and
   require exit 0; capture its JSON as a hash-bound receipt input without
   exposing credentials. Never run the checker against `alt` from `sfx`.
8. Recompute on `alt`: carrier ancestry, the two-file non-runtime allowlist, the
   tracked-field allowlist, the deployed source subtree, policy bytes, the
   selected-policy constant, protocol bytes, all five instrument hashes,
   field-store identity, a fresh cache-generation id, and the equal
   before/after delivery-configuration digests.
9. Publish `alt-deployment.json` on `alt` at `$RECEIPTS/alt-deployment.json`
   using `deployment-receipt-v2`, atomic rename, mode `0444`; then transfer only
   those immutable bytes to
   `…/operator-inputs/alt/alt-deployment.json` and re-verify length and SHA-256.

### Accrual and rollback (alt)

Accrual follows the same projection, stopping rule and identity construction as
`sfx`, computed on `alt`'s own traffic only. Close and hash `alt`'s store
snapshot and AE archive, publish `alt-accrual.json`
(`outcome-blind-accrual-receipt-v1`) on `alt`, then transfer only the immutable
receipt and the separately approved sealed bytes to their prebound external
input paths and re-verify lengths and SHA-256 before manifest construction.

Rollback follows the `sfx` procedure with `alt`'s own unit scope, restoring only
the exact values in `rollback-binding-v1`, restarting exactly once, and
verifying the new boot id, health, restored database hash, served protocol and
delivery digest **on `alt` itself**. Publish `alt-rollback.json`
(`rollback-receipt-v1`) on `alt` before transferring those immutable bytes to
the accepted-copy path.

### Rejecting an alt receipt

Reject a receipt that was produced, rewritten or completed off `alt`; that
carries a stale timestamp, a wrong protocol hash, missing raw provenance digests
or an unresolved path template; that was created before the operation it claims;
that leaves install mode ambiguous, or reports a non-editable package that was
not reinstalled, or a checkout-only proof with no served-protocol proof, or
identical boot ids, or `restart_count` other than 1, or a non-fresh cache
generation; or that shows any delivery-configuration drift, the wrong store,
source/policy/instrument drift, outcome access, an automated `alt` action, or an
incomplete field.

## Receipt schemas

Five receipt kinds, each **produced on the host it describes**, each written
once. Receipts are external inputs: they live at the paths below and are never
copied under the repository worktree.

| Receipt | Schema id | `sfx` path | `alt` producer path (then accepted copy) |
| --- | --- | --- | --- |
| Store preparation | `store-preparation-receipt-v2` | `…/field/$NS/receipts/sfx-store-preparation.json` | `${HOME}/…/field/$NS/alt/receipts/alt-store-preparation.json` → `…/field/$NS/operator-inputs/alt/alt-store-preparation.json` |
| Candidate promotion | `candidate-promotion-receipt-v1` | `…/field/$NS/receipts/sfx-candidate-promotion.json` | — (promotion is bound in the `alt` deployment receipt) |
| Deployment | `deployment-receipt-v2` | `…/field/$NS/receipts/sfx-deployment.json` | `…/alt/receipts/alt-deployment.json` → `…/operator-inputs/alt/alt-deployment.json` |
| Accrual | `outcome-blind-accrual-receipt-v1` | `…/field/$NS/receipts/sfx-accrual.json` | `…/alt/receipts/alt-accrual.json` → `…/operator-inputs/alt/alt-accrual.json` |
| Rollback | `rollback-receipt-v1` | `…/field/$NS/receipts/sfx-rollback.json` | `…/alt/receipts/alt-rollback.json` → `…/operator-inputs/alt/alt-rollback.json` |

`$NS` is `directional-zsum-r1-fresh-field-20260823`; the `sfx` prefix is
`/home/sfx/.local/share/living-memory`.

### Rules common to every receipt

* **Unknown fields reject.** The required-field list of a schema is exhaustive.
* **Regular file, never a symlink**, at an absolute host path — never a
  repo-relative one.
* **Atomic write:** same-directory temporary file → `fsync` → `rename`, then
  `chmod 0444`. Write once.
* **Immutable after publish.** A writable replacement, a rewrite, or a second
  receipt with the same semantics and different bytes all reject.
* **Byte equality on transfer.** For `alt`, the operator moves the *already
  immutable* bytes to the prebound accepted-copy path; the copy must match the
  producer's byte length and SHA-256 exactly. Re-verify at the destination.
* **Binding.** The consumer hashes the raw receipt bytes *before parsing*,
  validates the schema id and producer identity, then binds path, producer path
  where applicable, byte length, SHA-256, producer host, production instant and
  the protocol revision the receipt attests, in the next tracked `protocol.json`
  or `manifest.json` revision.
* **`success_conflict_value` is `null`.** A successful run records
  `protocol_conflict: null`, not `false` and not an empty string.
* **Reject** a receipt produced on the wrong host, with a hash mismatch, or
  containing any outcome field its schema forbids.

### Required fields

**`store-preparation-receipt-v2`** — `schema_id`, `accepted`, `producer_host`,
`produced_on_host`, `produced_at_utc`, `protocol_id`, `protocol_git_commit`,
`protocol_git_blob`, `protocol_bytes`, `protocol_bytes_sha256`,
`source_binding_id`, `origin_observation_realpath`, `origin_observation_bytes`,
`origin_observation_sha256`, `authoritative_source_realpath`,
`authoritative_source_bytes`, `authoritative_source_sha256`,
`source_paths_byte_equal`, `source_wal_bytes`, `source_open_handle_absent`,
`source_provenance_evidence_sha256`, `source_invariants_match`,
`source_host_configuration`, `source_snapshot_realpath`,
`source_snapshot_bytes`, `source_snapshot_sha256`, `field_store_realpath`,
`field_store_bytes`, `field_store_sha256`, `sqlite_integrity_ok`,
`inherited_foreign_key_baseline`, `recall_map_column_state`,
`pre_transform_nonnull_recall_map_rows`,
`post_transform_nonnull_recall_map_rows`, `transform_statement_sha256`,
`closure_schema_sha256_source`, `closure_schema_sha256_field`,
`closure_combined_sha256_source`, `closure_combined_sha256_field`,
`closure_table_counts`, `closure_table_sha256`, `organic_closure_equal`,
`organic_canonical_json_sha256`, `organic_expectations_match`,
`sealed_instrument_checks`, `evaluator_changed`, `preregistration_changed`,
`candidate_process_opened_store`, `outcome_accessed`, `protocol_conflict`.

`inherited_foreign_key_baseline` itself requires `schema_id`, `source`, `field`,
`source_field_metadata_equal`, `source_field_count_equal`,
`source_field_digest_equal`, `source_field_equal`, `new_violations`,
`source_closure_violations`, `field_closure_violations`, `closure_violations`;
each of `source` and `field` requires `metadata_user_table_count`,
`metadata_declaration_count`, `metadata_serialized_bytes`, `metadata_sha256`,
`tuple_cursor_metadata_sha256`, `tuple_count`, `tuple_serialized_bytes`,
`tuples_sha256`, `closure_violation_count`.

*Accepted only when:* `accepted=true`; `source_binding_id` is exactly
`sfx-mapdemo-snap-9f00abf4-v1`; both bound source paths recheck to 514121728 /
`9f00abf4…` and are byte-equal; `source_wal_bytes=0`;
`source_open_handle_absent=true`; provenance evidence matches;
`source_invariants_match=true`; field and seed bytes remain exact; integrity
succeeds; pre- and post-transform non-NULL `recall_map` rows are 0; closure
digests match with `organic_closure_equal=true`; the foreign-key baseline
reports the bound 35-tuple source expectation, all four equality booleans true,
`new_violations=0`, both closure counts 0 and `closure_violations=0`; the
1154-byte organic digest is exact; `organic_expectations_match=true`;
`evaluator_changed=false`; `preregistration_changed=false`;
`candidate_process_opened_store=false`; `outcome_accessed=false`;
`protocol_conflict=null`. A nonzero inherited global `tuple_count` is accepted
only at the exact bound count and digests.

**`candidate-promotion-receipt-v1`** — `schema_id`, `producer_host`,
`produced_at_utc`, `protocol_id`, `authorization_protocol_git_commit`,
`authorization_protocol_bytes_sha256`, `shared_checkout_realpath`,
`before_commit`, `after_commit`, `fast_forward_only`, `worktree_clean_before`,
`worktree_clean_after`, `carrier_tree`, `carrier_relationship_checks`,
`exact_nonruntime_allowlist_checks`, `tracked_field_allowlist_checks`,
`source_policy_instrument_checks`, `service_restarted`, `outcome_accessed`.

*Accepted only when:* the external operator alone fast-forwarded a clean shared
checkout from the observed base to the exact authorized carrier; every carrier
check is exact; `service_restarted=false`; `outcome_accessed=false`.

**`rollback-binding-v1`** (embedded in the deployment receipt) — `host_id`,
`captured_at_utc`, `predeployment_host_configuration_sha256`,
`predeployment_carrier_commit`, `predeployment_carrier_tree`,
`predeployment_checkout_root`, `predeployment_install_mode`,
`predeployment_distribution_location`, `predeployment_effective_db_realpath`,
`predeployment_db_bytes`, `predeployment_db_sha256`, `predeployment_store_closed`,
`predeployment_store_parts`, `rollback_material_root_realpath`,
`predeployment_rollback_bundle_realpath`, `predeployment_rollback_bundle_bytes`,
`predeployment_rollback_bundle_sha256`, `predeployment_service_unit`,
`predeployment_exec_start_argv`, `predeployment_boot_id`,
`predeployment_delivery_configuration_sha256`, `restore_material_available`,
`operator_only`.

*Accepted only when:* every path is host-resolved and non-secret;
`rollback_material_root_realpath` equals that host's prebound non-nested
rollback root; the binding was captured **after** stopping the old process and
**before** starting the candidate; the bundle covers the database main file plus
every present sidecar, closed and byte-for-byte; `predeployment_store_closed=true`;
`restore_material_available=true`; `operator_only=true`; and the delivery digest
equals `host_configuration_before.delivery_channel.before_canonical_sha256`.

**`deployment-receipt-v2`** — `schema_id`, `producer_host`, `produced_on_host`,
`produced_at_utc`, `protocol_id`, `protocol_bytes_sha256`,
`rollout_authorization_protocol_bytes_sha256`,
`store_preparation_receipt_bytes_sha256`, `carrier_commit`, `carrier_tree`,
`carrier_relationship_checks`, `host_configuration_before`,
`host_configuration_after`, `rollback_binding`, `rollback_ready`,
`delivery_configuration_unchanged`, `field_store_realpath`,
`initial_store_bytes`, `initial_store_sha256`, `before_boot_id`,
`after_boot_id`, `boot_ids_distinct`, `process_started_at_utc`,
`candidate_deployment_instant_utc`, `pid`, `effective_argv`,
`served_checkout_root`, `served_protocol_check_exit`,
`served_protocol_check_json_sha256`, `served_source_subtree_git_tree`,
`served_source_subtree_typed_content_sha256`, `served_policy_bytes_sha256`,
`served_selected_policy_sha256`, `cache_generation_id`,
`cache_generation_fresh`, `ae_journal_producer_roots`,
`ae_journal_archive_root`, `sealed_instrument_checks`, `restart_count`,
`outcome_accessed`.

*Accepted only when:* the exact authorized store and carrier; a valid
`rollback-binding-v1` with `rollback_ready=true`; distinct boot ids;
`restart_count=1`; served checker exit 0; exact source and policy hashes; both
non-runtime allowlist files exact; a fresh cache generation; unchanged delivery
configuration; all five instruments exact; `outcome_accessed=false`.

`host_configuration_before` / `_after` follow `host-configuration-v1`:
`captured_at_utc`, `host_id` (`sfx`|`alt`), `hostname`,
`service_manager_scope` (`user`|`system`), `service_unit`, `tls_unit`,
`unit_fragment_paths`, `unit_fragment_bytes_sha256`, `exec_start_argv`,
`environment_file_paths`, `environment_file_sanitized_sha256`,
`effective_db_path`, `config_file_path`, `config_file_bytes_sha256`,
`default_scope`, `transport` (`http`|`sse`), `bind_host`, `port`, `tls_enabled`,
`python_executable`, `console_script_path`, `install_mode`
(`editable`|`copied-package`), `editable_project_location`,
`installed_distribution_location`, `checkout_root`, `checkout_commit`,
`checkout_tree`, `source_subtree_git_tree`,
`source_subtree_typed_content_sha256`, `policy_bytes_sha256`,
`selected_policy_sha256`, `lm_environment`, `secret_presence`,
`delivery_channel`. Only `tls_unit`, `config_file_path`,
`config_file_bytes_sha256` and `editable_project_location` may be null, and only
when genuinely absent. `delivery_channel` requires `lm_url`,
`node_journal_producer_root`, `chat_journal_producer_roots`,
`AE_STREAM_PROBE`, `AE_STREAM_JOURNAL`, `AE_STREAM_JOURNAL_DIR`, `STATE_DIR`,
`AE_STREAM_INJECT_SILENCE_TOOLS`, `AE_STREAM_INJECT_CAP`, `AE_CHAT_LM_PROBE`,
`AE_CHAT_OPERATOR_MESSAGE_MAP`, `AE_CHAT_INJECT_MIN_TOOLS`,
`before_canonical_sha256`, `after_canonical_sha256`, `unchanged` — with
`before_canonical_sha256 == after_canonical_sha256` and `unchanged=true`.
Configurations are canonicalized as the SHA-256 of UTF-8 compact sorted-key JSON
after secret-value redaction and absolute realpath resolution.

**`outcome-blind-accrual-receipt-v1`** — `schema_id`, `producer_host`,
`produced_on_host`, `produced_at_utc`, `protocol_id`,
`preseal_protocol_git_commit`, `preseal_protocol_bytes_sha256`,
`deployment_receipt_bytes_sha256`, `candidate_deployment_instant_utc`,
`candidate_cutoff_utc`, `fixed_as_of_utc`, `closed_store_snapshot_realpath`,
`closed_store_snapshot_bytes`, `closed_store_snapshot_sha256`,
`closed_ae_journal_archive_realpath`, `closed_ae_journal_archive_bytes`,
`closed_ae_journal_archive_sha256`, `structural_inventory_schema_id`,
`structural_inventory_sha256`, `structural_stopping_rule_reproduced`,
`map_events`, `map_items`, `map_transport_sessions`, `injections_total`,
`injections_scorable`, `scorable_share`, `full_horizon_hours`,
`all_admitted_complete`, `earliest_nonnull_recall_map_at_or_after_deployment`,
`organic_closure_equal`, `sealed_instrument_checks`, `outcome_accessed`,
`protocol_conflict`.

*Accepted only when:* the inventory used only the outcome-blind projection;
`candidate_cutoff_utc` is the first qualifying structural boundary;
`fixed_as_of_utc` equals `candidate_cutoff_utc + 24h`;
`structural_stopping_rule_reproduced=true`; `map_events>=20`, `map_items>=20`,
`map_transport_sessions>=10`, `injections_scorable>=20`, `scorable_share>=0.50`,
`full_horizon_hours>=24`; every admitted unit is complete; the earliest map is
not before deployment; the organic closure remains exact; all instruments match;
`outcome_accessed=false`; `protocol_conflict=null`.

**`rollback-receipt-v1`** — `schema_id`, `producer_host`, `produced_on_host`,
`produced_at_utc`, `protocol_id`, `trigger`, `deployment_receipt_bytes_sha256`,
`rollback_binding_sha256`, `candidate_service_stopped_at_utc`,
`restored_carrier_commit`, `restored_carrier_tree`, `restored_checkout_root`,
`restored_install_mode`, `restored_effective_db_realpath`, `restored_db_bytes`,
`restored_db_sha256`, `restored_store_parts`, `restored_rollback_bundle_sha256`,
`before_rollback_boot_id`, `after_rollback_boot_id`, `boot_ids_distinct`,
`restart_count`, `health_ok`, `served_protocol_check_exit`,
`restored_delivery_configuration_sha256`, `delivery_configuration_restored`,
`candidate_inputs_preserved_read_only`, `candidate_cohort_void`,
`outcome_accessed`, `protocol_conflict`.

*Accepted only when:* produced on the rolled-back host; binds the exact
deployment receipt and rollback binding; restores every bound previous
carrier/install/configuration value plus the exact closed main-file and sidecar
bundle; changes the boot id with `restart_count=1`; passes health and
served-protocol verification; preserves candidate inputs read-only; marks the
cohort void; `outcome_accessed=false`; `protocol_conflict=null`.

## Validity criteria

**Outcome-independent cutoff.** `candidate_cutoff_utc` and `fixed_as_of_utc` are
chosen *only* from structural exposure and clock completion, by the stopping
rule above, before any outcome is readable. Consumption presence, count,
linkage, distance, rate, decision or verdict may never enter the choice — nor
may candidate query, result, node, label, hint, thinking, probe, curtail or
payload content, nor the other host's traffic, cutoff or outcome.

**Structural floors and horizon**, per host, on the complete prefix:

| Floor | Value |
| --- | --- |
| map events | ≥ 20 |
| map items | ≥ 20 |
| distinct map transport sessions | ≥ 10 |
| scorable injections | ≥ 20 |
| scorable share (`scorable / delivered`) | ≥ 0.50 |
| horizon | the complete 24 hours, `fixed_as_of_utc = cutoff + 24h` |

Every admitted unit must have matured for the full horizon
(`all_admitted_complete=true`, `full_horizon_hours>=24`).

**Disjointness from train and eval.** Across the four source-qualified identity
domains (event, transport session, session, cache), every one of these
intersections must be exactly empty: train × eval; sfx × train; sfx × eval;
alt × train; alt × eval; sfx × alt. The historical train/eval reference is
reconstructed from only the immutable source bytes hash-bound by
`dataset-manifest.json` (93512 bytes,
`54ee3bcc9d77b99b7b2c057db3e45b747c92a18d14ccec69eb2903dfe7cbf308`) and the
frozen split inputs, using the same `source_identity` and the exact `cache_key`
implementation — opaque component summaries are not sufficient. Both hosts
deliberately share `source_identity=local` precisely so that raw identity reuse
across `sfx` and `alt` stays detectable; host id is recorded separately and is
never mixed into a key. Any nonzero intersection, incomplete reconstruction,
missing mandatory event or cache identity, or hash drift **rejects the
manifest**. Never drop a conflicting unit or host to manufacture disjointness.

**Earliest map strictly after deployment.** In each closed snapshot, the
earliest non-NULL `recall_map` must fall **strictly after** that host's exact
deployment instant — the `/admin/info` `started_at` of the new boot is
`candidate_deployment_instant_utc`, and it is strictly earlier than every
admitted candidate event (`policy.json` `holdout_boundary.creation_rule`;
`cache_generation_semantics.generation_start`). The accrual-receipt boolean
`earliest_nonnull_recall_map_at_or_after_deployment` records this check; it is
true only when the strict inequality holds. A map event at or before the
deployment instant means the store was not fresh — stop.

**The unchanged instruments.** Verify byte size and SHA-256 before each store
proof, each deployment receipt, the final sealing, and every outcome invocation.
A mismatch is a protocol conflict, not a warning.

| Instrument | Bytes | SHA-256 |
| --- | --- | --- |
| `artifacts/recall-map/prereg.json` | 33003 | `9eb6a170459bb68138a972b4ba76b76abb044cbb00024cdb04e38e038a85b1b0` |
| `scripts/recall_map_effect.py` | 70157 | `e92aceeafb27ae736378f1b1866886cf8446b1cf17c1269a7cd6ad61a039eeea` |
| `scripts/recall_map_latency_bench.py` | 25652 | `f9f238f31a1e795a929ec8e97eb1d2090e8474beac0db4a07a8d3ea3767b8550` |
| `~/p/ae/docs/agent-stream-prereg.md` | 24211 | `68a3d4a597539a0a00d65d3dd792f3e2e6d734db263c0f79f086b0edbe98d9cb` |
| `~/p/ae/tools/lm_workload/stream_injection_metrics.py` | 58105 | `da9923fe37ad3c51c0c91fe6f3bb7e80083bcb17add9c0b983a6ced25ce7235d` |

`recall_map_latency_bench.py` is sealed but **not re-run in the field**: the
latency evidence is the pinned receipt `latency-after.json` — warm pooled p95
overhead 0.1958 against the 0.20 budget, measured on the same
`src/living_memory` tree and typed-content SHA-256 as the deployed candidate.
Re-running it, or replacing that pinned number, is a protocol conflict. The
paired-query (0.8526) and cold (0.2251) figures are disclosed diagnostics and
are not the preregistered endpoint.

**Running the instruments** — only after `release_authorized=true` is committed,
per host, never pooled:

```sh
cd "$CHECKOUT"            # the authorized checkout, at the sealed carrier
HOST=sfx                  # then repeat with HOST=alt; never one run over both
SNAP='<that host'"'"'s closed store snapshot>'
ASOF='<that host'"'"'s fixed_as_of_utc>'

# one invocation per host, written OUTSIDE the repository first
PYTHONPATH=src python3 scripts/recall_map_effect.py \
    --db "$SNAP" --as-of "$ASOF" --out "$NSROOT/tools/map-effect-$HOST.json"

# the AE ladder, per host, straight to its own tracked output
python3 /home/sfx/p/ae/tools/lm_workload/stream_injection_metrics.py \
    '<that host'"'"'s archived journal files>' \
    --output "artifacts/recall-map/relevance/field/injection-$HOST.json"
```

`tracked_outputs` names a single `map-effect.json`, but the measurement is
per-host and is never pooled. Assemble that one tracked file from the two
scratch reports **verbatim**, under separate `sfx` and `alt` keys: copy each
report as produced, and never sum, average, concatenate or otherwise merge
counts, rates or verdicts across the two. `injection-sfx.json` and
`injection-alt.json` are already per-host and are written directly.

**How to read the result.** The decision is made by two readings, both required
on **both** hosts independently:

1. `verdict.consumption.verdict == "PASS"` from `recall_map_effect.py`, at the
   unchanged sealed bar — `max(0.02, 0.5 × 0.4662) = 0.233` — with at least 20
   map events, 20 map items and 10 transport sessions.
2. `decision.verdict == "keep"` from `stream_injection_metrics.py`, with valid
   evidence, at least 20 scorable injections, scorable share at least 0.50, and
   every registered downgrade and noise rule preserved.

**The positional endpoint is reported separately and never decides.** Record
`verdict.positional.verdict` and `verdict.overall` verbatim in `verdict.md`
alongside the consumption verdict; do not let `overall` override, mask or
substitute for the consumption reading, and do not re-run the tool with
`--no-positional` to make the overall line look cleaner.

**An `INCONCLUSIVE` arm is never masked.** `recall_map_effect.py` returns
`INCONCLUSIVE` when the reference cohort is too small or the comparison is void;
`stream_injection_metrics.py` returns `inconclusive` below 20 scorable
injections or below 0.50 scorable share, and `void` when a validity
precondition fails. Report each verbatim. `INCONCLUSIVE` is never a pass, never
rewritten as a FAIL, never dropped from the report, and never repaired by
extending the window, lowering a floor, pooling hosts or re-measuring at a more
convenient as-of. A host whose evidence is missing is reported missing.

## Prohibitions

Each of these permanently voids the candidate holdout. On any of them, record
the violation and escalate — do not continue.

* **No pooling.** `sfx` and `alt` traffic, cutoffs, cohorts, journals and
  verdicts are never merged, averaged, substituted or dropped. Success requires
  two independently valid hosts under one unmodified protocol. One host's
  rollback neither supplies evidence for nor changes membership on the other.
* **No reused identities.** Zero overlap across train, eval and both candidate
  hosts in all four source-qualified identity domains. Never drop a conflicting
  unit or a host to manufacture disjointness; never mix host id into a
  source-qualified key.
* **No cutoff chosen from a favourable outcome.** The first qualifying
  structural boundary stands. Never extend, stop, include or exclude on
  consumption presence, rates, decisions or verdicts.
* **No threshold, instrument or delivery channel touched.** No preregistration
  change, no instrument change, no policy refit, no host-specific policy, no
  budget expansion (map ≤ 700 characters, instructions ≤ 150, injection render
  ≤ 400), no delivery-configuration change of any kind — triggers, renderers,
  chat paths, caps, silence rules, probes, endpoints, scope and wire budgets all
  stay byte-identical, with 0 silence violations and 0 cap violations. Payload
  changes stay additive, and every dropped candidate class and count stays
  attributable in the map payload *and* in the persisted
  `recall_events.recall_map` value — no silent caps.
* **No fabricated receipt, no substituted provenance.** Never author, complete,
  re-date or re-hash a receipt off the host it describes; never claim another
  source, snapshot or store as the bound one; never treat a path-only match as
  proof. **A missing host's evidence is reported missing** — a run with one
  valid host is a run with one valid host, not a pass.
* **While the embargo is active, do not** `SELECT`, dump, print, tail, grep,
  parse, join or count candidate queries, result lists, later follow-up
  contents, delivery-history outcomes or consumption state; do not open or parse
  any AE record kind or field outside the allowlisted `inject`/`session`
  projection; do not inspect even the *presence* of a `consume` record; do not
  run `recall_map_effect.py` at a candidate as-of or
  `stream_injection_metrics.py` over field journals; do not refit, tune, branch
  or gate the policy from any field observation; and do not use a health check,
  row count, cache hit or operator report as a proxy outcome.
* **Never open either bound source path through `MemoryStore`**, and never
  rebuild a store through SQLite backup, dump/restore or application APIs. The
  historical prior `MemoryStore` opening recorded in the provenance is bound
  history, not a repeatable step.
* **Never write anything except the six tracked outputs into the repository.**
  Stores, seeds, snapshots, journal archives, helper scripts and every receipt
  stay outside the worktree.
