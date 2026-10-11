"""Thread-safe per-customer embedding memory."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np

from .embed import HashingEmbedder


@dataclass
class Event:
    event_id: str
    customer_id: str
    text: str
    ts: float
    kind: str = "message"

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        for k in ("event_id", "customer_id", "text", "ts"):
            if k not in d:
                raise ValueError(f"missing field {k}")
        if not str(d["text"]).strip():
            raise ValueError("empty text")
        return cls(str(d["event_id"]), str(d["customer_id"]), str(d["text"]),
                   float(d["ts"]), str(d.get("kind", "message")))


@dataclass
class _CustomerMem:
    vecs: np.ndarray
    events: list = field(default_factory=list)
    n: int = 0


class MemoryStore:
    """Holds, per customer, a growable matrix of unit vectors plus the events.

    search() scores = cosine + recency_weight * exp(-age / half_life-ish).
    Max `max_per_customer` items; oldest are evicted first.
    """

    def __init__(self, embedder=None, max_per_customer: int = 2000, recency_weight: float = 0.0,
                 recency_tau: float = 3600.0):
        self.embedder = embedder or HashingEmbedder()
        self.max = max_per_customer
        self.recency_weight = recency_weight
        self.recency_tau = recency_tau
        self._mem: dict[str, _CustomerMem] = {}
        self._seen: set[str] = set()
        self._lock = threading.RLock()
        self.ingested = 0
        self.duplicates = 0

    def add(self, ev: Event) -> bool:
        vec = self.embedder.embed(ev.text)
        with self._lock:
            if ev.event_id in self._seen:
                self.duplicates += 1
                return False
            self._seen.add(ev.event_id)
            m = self._mem.get(ev.customer_id)
            if m is None:
                m = self._mem[ev.customer_id] = _CustomerMem(np.zeros((16, self.embedder.dim), np.float32))
            if m.n == self.max:  # evict oldest
                m.vecs[:-1] = m.vecs[1:]
                m.events.pop(0)
                m.n -= 1
            if m.n == len(m.vecs):
                m.vecs = np.vstack([m.vecs, np.zeros_like(m.vecs)])[: max(self.max, 16)]
            m.vecs[m.n] = vec
            m.events.append(ev)
            m.n += 1
            self.ingested += 1
            return True

    def count(self, customer_id: str) -> int:
        with self._lock:
            m = self._mem.get(customer_id)
            return m.n if m else 0

    def customers(self) -> int:
        with self._lock:
            return len(self._mem)

    def search(self, customer_id: str, query: str, k: int = 5, now: float | None = None,
               recency_weight: float | None = None) -> list[tuple[Event, float]]:
        q = self.embedder.embed(query)
        rw = self.recency_weight if recency_weight is None else recency_weight
        with self._lock:
            m = self._mem.get(customer_id)
            if not m or m.n == 0:
                return []
            scores = m.vecs[: m.n] @ q
            events = list(m.events)
        if rw:
            now = now if now is not None else max(e.ts for e in events)
            ages = np.array([max(0.0, now - e.ts) for e in events], dtype=np.float32)
            scores = scores + rw * np.exp(-ages / self.recency_tau)
        k = min(k, len(events))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(events[i], float(scores[i])) for i in idx]

    def search_fresh(self, customer_id: str, query: str, k: int = 5, pool: int = 10,
                     tie: float = 0.85) -> list[tuple[Event, float]]:
        """Recency-aware rerank: take the top `pool` by cosine, keep those within
        `tie` x best score, and put the newest of those first. Old facts that were
        superseded by a similar, newer statement drop below it."""
        cand = self.search(customer_id, query, pool, recency_weight=0)
        if not cand:
            return []
        best = cand[0][1]
        near = sorted([c for c in cand if c[1] >= tie * best], key=lambda c: -c[0].ts)
        rest = [c for c in cand if c[1] < tie * best]
        return (near + rest)[:k]

    def recent(self, customer_id: str, n: int) -> list[Event]:
        with self._lock:
            m = self._mem.get(customer_id)
            return list(m.events[-n:]) if m else []


def percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    if not s:
        return math.nan
    i = (len(s) - 1) * p / 100
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)
