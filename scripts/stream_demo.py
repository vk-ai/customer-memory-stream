"""End-to-end streaming check against a REAL Kafka-compatible broker.

Publishes the eval conversation events to `customer-events`, starts the
StreamIngestor, waits until all are stored, then queries the HTTP API.
Writes results/stream_demo.json.  usage: KAFKA_BOOTSTRAP=localhost:9092 python scripts/stream_demo.py
"""
import json, os, sys, time, uuid
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from cms.api import create_app
from cms.memeval import generate
from cms.memory import MemoryStore, percentile
from cms.stream import StreamIngestor, make_producer, publish

bs = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
n_cust = int(os.environ.get("N_CUSTOMERS", "100"))
topic = f"customer-events-{uuid.uuid4().hex[:6]}"
events, cases = generate(n_cust, seed=1)
store = MemoryStore(recency_weight=0.15)
ing = StreamIngestor(store, bs, topic=topic, group_id=f"g-{topic}", from_beginning=True)
ing.start()
time.sleep(4)  # let the consumer join the group before we publish, so lag = live lag
prod = make_producer(bs)
t0 = time.time()
for e in events:
    publish(prod, e.__dict__, topic)
publish(prod, {"event_id": "bad-1", "customer_id": "x"}, topic)  # malformed on purpose
prod.flush()
t_pub = time.time() - t0
t1 = t0
while (store.ingested < len(events) or ing.bad < 1) and time.time() - t1 < 120:
    time.sleep(0.05)
t_ing = time.time() - t1
ing.stop()
client = TestClient(create_app(store, bootstrap=""))
c = next(x for x in cases if x.slot == 'order')
hits = client.get(f"/customers/{c.customer_id}/memory", params={"q": c.question, "k": 3}).json()["hits"]
res = {"broker": bs, "events": len(events), "stored": store.ingested, "bad_messages_dropped": ing.bad,
       "publish_s": round(t_pub, 2), "publish_start_to_all_indexed_s": round(t_ing, 2),
       "events_per_s_ingest": round(store.ingested / t_ing),
       "broker_to_store_lag_ms_p50": round(percentile(ing.ingest_lag_ms, 50), 1),
       "broker_to_store_lag_ms_p95": round(percentile(ing.ingest_lag_ms, 95), 1),
       "sample_query": c.question, "sample_top_hit": hits[0]["text"] if hits else None}
os.makedirs("results", exist_ok=True)
json.dump(res, open("results/stream_demo.json", "w"), indent=2)
print(json.dumps(res, indent=2))
assert store.ingested == len(events) and ing.bad == 1, "stream path incomplete"
