import cProfile, pstats, sys, time, json, shutil, os
sys.path.insert(0, sys.argv[1] + "/src")
from living_memory.storage import MemoryStore
from living_memory.config import MemoryConfig, load_config
from living_memory.consolidation import memory_consolidate
db = sys.argv[2]; scope = sys.argv[3]; out = sys.argv[4]
os.environ.setdefault("LM_AUTO_CONSOLIDATE_POLICY", "adaptive")
cfg = load_config(os.path.expanduser("~/.config/living-memory/config.toml"))
from dataclasses import replace
cfg = replace(cfg, db_path=db)
store = MemoryStore(cfg) if not hasattr(MemoryStore, "open") else MemoryStore.open(cfg)
prof = cProfile.Profile()
t0 = time.perf_counter()
prof.enable()
res = memory_consolidate(store, scope=scope, force=False)
prof.disable()
dt = time.perf_counter() - t0
prof.dump_stats(out + ".prof")
summary = {
  "seconds": dt,
  "clusters_considered": res.clusters_considered,
  "traces_considered": res.traces_considered,
  "created": sorted(n.id for n in res.concepts_created),
  "updated": sorted(n.id for n in res.concepts_updated),
  "promoted": sorted(n.id for n in res.concepts_promoted),
  "schemas_created": sorted(n.id for n in res.schemas_created),
  "schemas_updated": sorted(n.id for n in res.schemas_updated),
  "decayed": sorted(n.id for n in res.decayed),
}
json.dump(summary, open(out + ".json", "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in summary.items()}))
pstats.Stats(out + ".prof").sort_stats("cumulative").print_stats(40)
