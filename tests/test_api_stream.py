import json
import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from cms.api import create_app
from cms.memory import MemoryStore
from cms.stream import StreamIngestor


def test_api_ingest_and_search():
    c = TestClient(create_app(MemoryStore(), bootstrap=""))
    assert c.get("/healthz").json() == {"ok": True, "kafka": False}
    r = c.post("/events", json={"event_id": "1", "customer_id": "c", "text": "Reach me by WhatsApp only", "ts": 1})
    assert r.status_code == 202 and r.json()["added"]
    assert not c.post("/events", json={"event_id": "1", "customer_id": "c", "text": "dup", "ts": 2}).json()["added"]
    hits = c.get("/customers/c/memory", params={"q": "preferred contact channel whatsapp", "k": 3}).json()["hits"]
    assert hits[0]["text"].startswith("Reach me")
    assert c.get("/stats").json()["duplicates"] == 1


def test_api_validation():
    c = TestClient(create_app(MemoryStore(), bootstrap=""))
    assert c.get("/customers/c/memory").status_code == 422
    assert c.get("/customers/c/memory", params={"q": "x", "k": 999}).status_code == 422
    assert c.post("/events", json={"event_id": "1", "customer_id": "c", "text": "  ", "ts": 1}).status_code == 422


def test_ingestor_handle_without_broker():
    s = MemoryStore()
    ing = StreamIngestor(s, "unused:9092")
    assert ing.handle(json.dumps({"event_id": "1", "customer_id": "c", "text": "hi there", "ts": 1}).encode(), 1)
    assert not ing.handle(b"not json") and not ing.handle(b'{"event_id": "x"}')
    assert ing.bad == 2 and s.ingested == 1


KAFKA = os.getenv("KAFKA_BOOTSTRAP")


@pytest.mark.skipif(not KAFKA, reason="needs a real Kafka/Redpanda broker (KAFKA_BOOTSTRAP)")
def test_real_broker_roundtrip():
    from cms.stream import make_producer, publish
    topic = f"t-{uuid.uuid4().hex[:8]}"
    s = MemoryStore()
    ing = StreamIngestor(s, KAFKA, topic=topic, group_id=f"g-{topic}").start()
    p = make_producer(KAFKA)
    for i in range(50):
        publish(p, {"event_id": str(i), "customer_id": f"c{i % 5}", "text": f"order {i} shipped", "ts": i}, topic)
    publish(p, {"event_id": "50", "customer_id": "c0", "text": "My order number is Z-1", "ts": 99}, topic)
    p.flush()
    deadline = time.time() + 60
    while s.ingested < 51 and time.time() < deadline:
        time.sleep(0.1)
    ing.stop()
    assert s.ingested == 51 and s.customers() == 5
    assert "Z-1" in s.search("c0", "order number", 1)[0][0].text
