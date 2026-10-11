"""Run the memory eval and write results/memeval_<embedder>.json with Wilson 95% CIs.
usage: python scripts/run_eval.py [--minilm]"""
import json, math, sys
sys.path.insert(0, ".")
from cms.memeval import run


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


emb = None
if "--minilm" in sys.argv:
    from cms.embed import MiniLMEmbedder
    emb = MiniLMEmbedder()
r = run(embedder=emb)
meta = r.pop("_meta")
out = {"_meta": meta}
for name, m in r.items():
    out[name] = {}
    for key, v in m.items():
        n = meta["updated_facts"] if key.startswith("stale") else meta["questions"]
        out[name][key] = {"value": round(v, 3), "ci95": wilson(v, n)}
path = f"results/memeval_{meta['embedder']}.json"
json.dump(out, open(path, "w"), indent=2)
print(json.dumps(out, indent=2))
