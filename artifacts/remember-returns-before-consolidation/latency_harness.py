"""Due write + concurrent recalls against a corpus copy, in-process server.

usage: latency_harness.py <repo-with-src> <db> <scope> <out.json>
"""
import json, os, sys, threading, time
repo, db, scope, out = sys.argv[1:5]
# The hourly scope-agnostic decay sweep is not part of the consolidation pass;
# on a copy it fires inside the first write and retires traces, which moves
# the scope off its due count. Off for both arms.
os.environ["LM_DECAY_SWEEP_INTERVAL_SEC"] = "0"
LOAD = {"start": os.getloadavg()}
sys.path.insert(0, repo + "/src")
for line in open(os.path.expanduser("~/.config/living-memory/env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line and "TOKEN" not in line:
        k, v = line.split("=", 1); os.environ.setdefault(k, v)
from living_memory.server import create_mcp_server
import inspect, contextlib
from living_memory import consolidation as _C
STEP_HOLDS = []
if "guard" in inspect.signature(_C.memory_consolidate).parameters:
    _real = _C.memory_consolidate
    def _timed(store, **kw):
        g = kw.get("guard")
        if g is not None:
            @contextlib.contextmanager
            def timed_guard():
                with g():
                    t = time.perf_counter()
                    try:
                        yield
                    finally:
                        STEP_HOLDS.append(time.perf_counter() - t)
            kw["guard"] = timed_guard
        return _real(store, **kw)
    _C.memory_consolidate = _timed

class FakeMCP:
    def __init__(self, name, instructions=None):
        self.name, self.instructions, self.tools, self.resources, self.prompts = name, instructions, {}, {}, {}
    def tool(self, func=None, **kw):
        def d(inner): self.tools[str(kw.get("name") or inner.__name__)] = inner; return inner
        return d if func is None else d(func)
    def resource(self, uri, **kw):
        def d(f): self.resources[uri] = f; return f
        return d
    def prompt(self, func=None, **kw):
        def d(inner): self.prompts[str(kw.get("name") or inner.__name__)] = inner; return inner
        return d if func is None else d(func)

mcp = create_mcp_server(db, mcp_factory=FakeMCP, config_path=os.path.expanduser("~/.config/living-memory/config.toml"))
store = mcp.memory_store
def count():
    return int(store.connection.execute("SELECT COUNT(*) FROM nodes WHERE level='trace' AND scope=? AND decayed=0", (scope,)).fetchone()[0])
filler = 0
while count() % 100 != 99:
    store.append_trace(f"latency harness filler note {filler} about harbor crane maintenance", {"scope": scope, "agent": "harness"})
    filler += 1
QUERIES = ["consolidation pass latency", "goal tree decomposition", "recall ranking precision", "sqlite migration", "harbor crane maintenance"]
warm = []
for q in QUERIES:
    t = time.perf_counter(); mcp.tools["memory_recall"](q, scope=scope); warm.append(time.perf_counter() - t)
print("warm recalls", [round(x, 2) for x in warm], flush=True)
done = threading.Event(); samples = []
def recaller():
    i = 0
    while not done.is_set():
        q = QUERIES[i % len(QUERIES)]; i += 1
        t0 = time.time(); t = time.perf_counter()
        mcp.tools["memory_recall"](q, scope=scope)
        samples.append({"at": t0, "seconds": time.perf_counter() - t})
        done.wait(1.0)
pass_started = time.time()
th = threading.Thread(target=recaller, daemon=True); th.start()
time.sleep(3)  # recall baseline before the due write
before_write = time.time()
LOAD["write"] = os.getloadavg()
t = time.perf_counter()
resp = mcp.tools["memory_remember"]("latency harness due write: the hundredth trace of the scope", {"scope": scope, "agent": "harness"})
write_seconds = time.perf_counter() - t
print("write_seconds", round(write_seconds, 2), resp.get("auto_consolidation") if not isinstance(resp.get("auto_consolidation"), dict) or "status" in resp["auto_consolidation"] else "inline-summary", flush=True)
sched = getattr(mcp, "auto_consolidation", None)
if sched is not None and hasattr(sched, "wait_idle"):
    sched.wait_idle(timeout=7200)
    last = sched.last_result(scope)
    summary = last["summary"]; pass_seconds = last["duration_seconds"]
else:
    summary = resp["auto_consolidation"]; pass_seconds = write_seconds
pass_end = time.time()
LOAD["pass_end"] = os.getloadavg()
time.sleep(3)
done.set(); th.join(30)
during = [s["seconds"] for s in samples if before_write <= s["at"] <= pass_end]
outside = [s["seconds"] for s in samples if s["at"] < before_write or s["at"] > pass_end]
def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else None
touched = sorted(set(summary["concepts_created"]) | set(summary["concepts_updated"]))
nodes = []
for nid in touched:
    n = store.get_node(nid)
    nodes.append({"id": nid, "created": nid in summary["concepts_created"], "content": n.content,
                  "sources": sorted(store.get_node(s).content[:200] if store.get_node(s) else s for s in n.source_traces),
                  "cluster_key": n.provenance.get("cluster_key"), "confidence": n.confidence})
res = {"load": LOAD, "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}, "repo": repo, "scope": scope, "filler": filler, "write_seconds": write_seconds, "pass_seconds": pass_seconds,
       "response_auto_consolidation": resp.get("auto_consolidation") if sched is not None else "inline",
       "clusters_considered": summary["clusters_considered"], "traces_considered": summary["traces_considered"],
       "created": len(summary["concepts_created"]), "updated": len(summary["concepts_updated"]),
       "schemas_created": len(summary["schemas_created"]), "schemas_updated": len(summary["schemas_updated"]),
       "decayed": len(summary["decayed"]),
       "recall_during_pass": {"n": len(during), "p50": q(during, .5), "p95": q(during, .95), "max": max(during) if during else None},
       "recall_outside_pass": {"n": len(outside), "p50": q(outside, .5), "max": max(outside) if outside else None},
       "warm_recalls": warm,
       "guard_steps": {"n": len(STEP_HOLDS), "total": sum(STEP_HOLDS), "max": max(STEP_HOLDS) if STEP_HOLDS else None,
                       "top5": sorted(STEP_HOLDS)[-5:]}, "touched_nodes": nodes}
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k != "touched_nodes"}, ensure_ascii=False), flush=True)
