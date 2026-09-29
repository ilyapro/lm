"""Build the consolidation embedding cache on a corpus copy, then compare
reference vs indexed clustering on the same vectors (membership + timing)."""
import sys, time, json, os
from contextlib import nullcontext
sys.path.insert(0, sys.argv[1] + "/src")
from dataclasses import replace
from living_memory.storage import MemoryStore
from living_memory.config import load_config
from living_memory import consolidation as C
from living_memory.embeddings import LocalEmbeddingModel
db, scope, out = sys.argv[2], sys.argv[3], sys.argv[4]
cfg = replace(load_config(os.path.expanduser("~/.config/living-memory/config.toml")), db_path=db)
store = MemoryStore(cfg)
traces = store.list_nodes(level="trace", scope=scope, include_decayed=False, limit=C.DEFAULT_RECENT_LIMIT)
emb = LocalEmbeddingModel(model_name=store.config.embedding_model)
t = time.perf_counter(); vecs = C._trace_embeddings(store, traces, emb, nullcontext); t_embed = time.perf_counter() - t
print("embed(cold or warm)", len(traces), round(t_embed, 1), flush=True)
t = time.perf_counter(); vecs2 = C._trace_embeddings(store, traces, emb, nullcontext); t_embed2 = time.perf_counter() - t
print("embed(warm)", round(t_embed2, 2), flush=True)
prepared = []
for tr in traces:
    tokens = C._significant_tokens(tr.content); e = vecs.get(tr.id)
    if not tokens and e is None: continue
    prepared.append((tr, tokens, e))
t = time.perf_counter(); idx = C._cluster_traces_indexed(prepared, 384); t_idx = time.perf_counter() - t
print("indexed", len(idx), round(t_idx, 2), flush=True)
res = {"traces": len(traces), "embed_first": t_embed, "embed_warm": t_embed2, "indexed_s": t_idx, "indexed_clusters": len(idx)}
if "--reference" in sys.argv:
    t = time.perf_counter(); ref = C._cluster_prepared_reference(prepared); t_ref = time.perf_counter() - t
    print("reference", len(ref), round(t_ref, 1), flush=True)
    a = [[x.id for x in c.traces] for c in ref]; b = [[x.id for x in c.traces] for c in idx]
    res.update(reference_s=t_ref, reference_clusters=len(ref), identical_membership=(a == b),
               identical_strategy=[c.strategy for c in ref] == [c.strategy for c in idx],
               identical_keys=[C._cluster_key(c) for c in ref] == [C._cluster_key(c) for c in idx],
               max_sum_diff=max((max(abs(p - q) for p, q in zip(x.embedding_sum, y.embedding_sum)) for x, y in zip(ref, idx) if x.embedding_sum), default=0.0))
    print(res, flush=True)
json.dump(res, open(out, "w"), indent=1)
