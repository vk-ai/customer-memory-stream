"""Kafka/Redpanda consumer that feeds the MemoryStore (kafka-python client)."""
from __future__ import annotations

import json
import logging
import threading
import time

from .memory import Event, MemoryStore

log = logging.getLogger(__name__)
TOPIC = "customer-events"


def make_producer(bootstrap: str):
    from kafka import KafkaProducer
    return KafkaProducer(bootstrap_servers=bootstrap, key_serializer=lambda k: k.encode(),
                         value_serializer=lambda v: json.dumps(v).encode(), linger_ms=5, acks="all")


def publish(producer, ev: dict, topic: str = TOPIC):
    # keyed by customer so one customer's events stay ordered within a partition
    return producer.send(topic, key=ev["customer_id"], value=ev)


class StreamIngestor:
    """Consumes events and writes them into the store; tracks bad messages and lag."""

    def __init__(self, store: MemoryStore, bootstrap: str, topic: str = TOPIC,
                 group_id: str = "cms-memory", from_beginning: bool = True):
        self.store, self.bootstrap, self.topic = store, bootstrap, topic
        self.group_id, self.from_beginning = group_id, from_beginning
        self.bad = 0
        self.ingest_lag_ms: list[float] = []  # broker timestamp -> stored
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def handle(self, value: bytes, broker_ts_ms: int | None = None) -> bool:
        try:
            ev = Event.from_dict(json.loads(value))
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            self.bad += 1
            log.warning("dropping bad event: %s", e)
            return False
        ok = self.store.add(ev)
        if broker_ts_ms:
            self.ingest_lag_ms.append(time.time() * 1000 - broker_ts_ms)
        return ok

    def run(self):
        from kafka import KafkaConsumer
        c = KafkaConsumer(self.topic, bootstrap_servers=self.bootstrap, group_id=self.group_id,
                          auto_offset_reset="earliest" if self.from_beginning else "latest",
                          enable_auto_commit=True, consumer_timeout_ms=500)
        try:
            while not self._stop.is_set():
                for msg in c:
                    self.handle(msg.value, msg.timestamp)
                    if self._stop.is_set():
                        break
        finally:
            c.close()

    def start(self):
        self._thread = threading.Thread(target=self.run, daemon=True, name="cms-ingest")
        self._thread.start()
        return self

    def stop(self, timeout: float = 5):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
