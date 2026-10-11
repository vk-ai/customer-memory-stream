# customer-memory-stream

**Learning project** (personal OSS, not production and not tied to any employer): stream customer
events from Kafka/Redpanda into a per-customer embedding memory and serve it from a low-latency
FastAPI retrieval endpoint. Includes a latency load test, a multi-turn agent-memory eval, and a small
uplift (T-learner) module on synthetic data.

```
producer --(topic customer-events, key=customer_id)--> Kafka / Redpanda
                                                         |
                                     StreamIngestor (kafka-python consumer group)
                                     validate -> dedupe(event_id) -> embed -> MemoryStore
                                                         |
             FastAPI  GET /customers/{id}/memory?q=..&k=..   POST /events   GET /stats
```

- `cms/embed.py` - `HashingEmbedder` (default; signed feature hashing of words, bigrams, char trigrams,
  512-d, no download) and optional `MiniLMEmbedder` (all-MiniLM-L6-v2 via `transformers`).
- `cms/memory.py` - thread-safe per-customer numpy matrices, dedupe by `event_id`, oldest-first
  eviction, cosine search, optional recency boost, and `search_fresh` (top-10 by cosine, then newest
  first among near-ties so superseded facts drop below newer ones).
- `cms/stream.py` - producer helper and consumer that drops malformed events and records broker->store lag.
- `cms/memeval.py` - seeded multi-turn memory eval; `cms/uplift.py` - T-learner.

## Run

```bash
pip install -e ".[dev]"                     # add ".[neural]" for MiniLM
pytest -q                                   # broker test skips without KAFKA_BOOTSTRAP
docker compose up                           # Redpanda + API (see "not run" note below)
KAFKA_BOOTSTRAP=localhost:9092 python scripts/stream_demo.py
python scripts/run_eval.py [--minilm]
bash loadtest/run.sh 20 60                  # locust, headless
```

## Measured results (real runs, Oct 2026, 8-vCPU Linux box, CPU only, Python 3.12)

**Streaming against a real broker.** No Docker on the box, so I ran the Apache Kafka 4.3.1 binary
(KRaft, single node) locally. `scripts/stream_demo.py`, 100 customers / 6,038 events + 1 malformed
event (`results/stream_demo.json`):

| metric | value |
|---|---|
| events stored / malformed dropped | 6,038 / 1 |
| publish start -> all indexed | 0.87 s (~6.9k events/s) |
| broker timestamp -> stored, p50 / p95 | 442 ms / 561 ms |

The lag is for a **burst** (all 6k events published in 0.7 s, so they queue behind each other). It is not
the latency of a single event on a quiet topic. CI repeats this against an `apache/kafka:3.9.0`
service container on every push.

**Retrieval latency under load** (locust headless, single uvicorn worker, store preloaded with 12,065
events for 200 customers, hashing embedder, 90% search / 10% ingest; `results/locust_*.csv`):

| concurrent users | req/s | p50 | p95 | p99 | failures |
|---|---|---|---|---|---|
| 1 (30 s run) | 354 | 1 ms | 9 ms | 13 ms | 0 |
| 20 (60 s run) | 1,129 | 15 ms | 26 ms | 36 ms | 0 |

Locust and the server shared the same machine, and locust's percentiles are approximate.

**Multi-turn memory eval** (`results/memeval_*.json`): 200 synthetic customers, about 60 turns each,
12,065 events, 800 questions, and 318 facts that were later updated. Questions are worded differently
from the statements. Values are rates with Wilson 95% CIs; k = 3.

| strategy | hashing hit@1 | hashing hit@3 | MiniLM hit@1 | MiniLM hit@3 | MiniLM stale@1 on updated facts |
|---|---|---|---|---|---|
| last-5 turns, no retrieval | 0.006 | 0.031 | 0.006 | 0.031 | 0.000 |
| cosine retrieval | 0.376 [.343,.410] | 0.505 | 0.545 [.510,.579] | 0.695 | 0.355 |
| cosine + recency boost (w=0.15) | 0.305 | 0.409 | 0.554 | 0.661 | 0.223 |
| cosine top-10 -> newest near-tie | **0.394** [.360,.428] | 0.482 | **0.589** [.554,.622] | 0.682 | **0.198** |

Takeaways: retrieval is essential, since a recent-turns window almost never contains the fact. The neural
embedder adds about 17-20 points because questions are paraphrased. Recency-aware reranking roughly
halves "stale" answers, where the agent quotes an outdated value, at a small cost in hit@3. Even the best
setting misses about 40% at rank 1, so this is far from solved.

**Uplift (synthetic, clearly made up)** (`results/uplift_synthetic.json`): T-learner with gradient boosting
on 20k simulated customers with a known true effect. On held-out data: correlation between predicted
and true uplift 0.64, PEHE 0.098. Observed uplift in the model's top 20% is 0.181, against 0.072
overall and a true value of 0.173.

## Real vs stubbed

| part | status |
|---|---|
| Kafka consume/produce, consumer group, keyed partitions, malformed-event drop | **Real.** Run against Kafka 4.3.1 locally and apache/kafka 3.9.0 in CI |
| Redpanda via docker-compose | **Not run**: no Docker on the build box. The file is provided, but only the Kafka path was exercised |
| Embeddings | Hashing embedder is real but lexical. MiniLM is real but optional, not in CI, and has no batching |
| Memory store | In-process and in-memory only: no persistence, no sharding, and lost on restart; brute-force cosine (no ANN index) |
| Exactly-once / replay | No. Auto-commit offsets plus in-memory dedupe, so a restart loses the dedupe set and the memory |
| Auth, PII handling, retention/deletion | None |
| Eval data | Synthetic, templated conversations, not real customer transcripts. The answer is scored by string match on the retrieved memory; no LLM generates an answer |
| Load test | Single machine, single worker, synthetic queries. No network hops |
| Uplift | Synthetic data with a planted effect. It demonstrates the method only and says nothing about real customers |
