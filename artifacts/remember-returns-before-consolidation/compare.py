import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
def key(n):
    return n["id"] if not n["created"] else ("new", n["cluster_key"], tuple(n["sources"]))
ka = {key(n): n for n in a["touched_nodes"]}; kb = {key(n): n for n in b["touched_nodes"]}
same = set(ka) & set(kb)
out = {
  "before": {k: a[k] for k in ("clusters_considered", "traces_considered", "created", "updated", "schemas_created", "schemas_updated", "decayed")},
  "after": {k: b[k] for k in ("clusters_considered", "traces_considered", "created", "updated", "schemas_created", "schemas_updated", "decayed")},
  "touched_matched": len(same), "only_before": len(set(ka) - set(kb)), "only_after": len(set(kb) - set(ka)),
  "content_identical": sum(ka[k]["content"] == kb[k]["content"] for k in same),
  "sources_identical": sum(ka[k]["sources"] == kb[k]["sources"] for k in same),
  "confidence_identical": sum(abs(ka[k]["confidence"] - kb[k]["confidence"]) < 1e-9 for k in same),
}
print(json.dumps(out, indent=1))
