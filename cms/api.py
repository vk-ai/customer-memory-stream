"""FastAPI retrieval service."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .memory import Event, MemoryStore


class EventIn(BaseModel):
    event_id: str
    customer_id: str
    text: str
    ts: float
    kind: str = "message"


def create_app(store: MemoryStore | None = None, bootstrap: str | None = None) -> FastAPI:
    store = store or MemoryStore(recency_weight=float(os.getenv("CMS_RECENCY_WEIGHT", "0.15")))
    bootstrap = bootstrap if bootstrap is not None else os.getenv("KAFKA_BOOTSTRAP")
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app):
        if bootstrap:
            from .stream import StreamIngestor
            state["ingestor"] = StreamIngestor(store, bootstrap).start()
        yield
        if "ingestor" in state:
            state["ingestor"].stop()

    app = FastAPI(title="customer-memory-stream", lifespan=lifespan)
    app.state.store = store

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "kafka": bool(bootstrap)}

    @app.post("/events", status_code=202)
    def ingest(ev: EventIn):  # direct path for tests / no-broker mode
        try:
            added = store.add(Event.from_dict(ev.model_dump()))
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"added": added}

    # sync def -> runs in threadpool; search is numpy and holds the lock briefly
    @app.get("/customers/{customer_id}/memory")
    def search(customer_id: str, q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=50)):
        hits = store.search(customer_id, q, k)
        return {"customer_id": customer_id, "hits": [
            {"event_id": e.event_id, "text": e.text, "ts": e.ts, "kind": e.kind, "score": round(s, 4)}
            for e, s in hits]}

    @app.get("/stats")
    def stats():
        ing = state.get("ingestor")
        return {"customers": store.customers(), "ingested": store.ingested,
                "duplicates": store.duplicates, "bad_messages": ing.bad if ing else 0}

    return app


app = create_app()
