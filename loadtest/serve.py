"""Serve the API preloaded with the synthetic conversations (12k events, 200 customers)."""
import sys
sys.path.insert(0, ".")
import uvicorn
from cms.api import create_app
from cms.memeval import generate
from cms.memory import MemoryStore

store = MemoryStore(recency_weight=0.15)
events, _ = generate(200, seed=0)
for e in events:
    store.add(e)
app = create_app(store, bootstrap="")
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8077, log_level="warning")
