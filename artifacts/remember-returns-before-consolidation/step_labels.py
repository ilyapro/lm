import sys, os, time, contextlib, traceback, json
sys.path.insert(0, sys.argv[1] + "/src")
os.environ["LM_DECAY_SWEEP_INTERVAL_SEC"] = "0"
from dataclasses import replace
from living_memory.storage import MemoryStore
from living_memory.config import load_config
from living_memory.consolidation import memory_consolidate
cfg = replace(load_config(os.path.expanduser("~/.config/living-memory/config.toml")), db_path=sys.argv[2])
store = MemoryStore(cfg)
holds = []
@contextlib.contextmanager
def guard():
    frame = traceback.extract_stack(limit=4)[-3]
    t = time.perf_counter()
    try:
        yield
    finally:
        holds.append((round(time.perf_counter() - t, 3), frame.lineno, frame.line))
t = time.perf_counter()
r = memory_consolidate(store, scope=sys.argv[3], guard=guard)
total = time.perf_counter() - t
agg = {}
for s, ln, line in holds:
    a = agg.setdefault(f"{ln}: {line}", [0, 0.0, 0.0]); a[0] += 1; a[1] += s; a[2] = max(a[2], s)
print(json.dumps({"total": total, "held": sum(h[0] for h in holds), "by_site": agg}, indent=1))
