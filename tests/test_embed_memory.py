import numpy as np
import pytest
from cms.embed import HashingEmbedder
from cms.memory import Event, MemoryStore, percentile


def test_embed_unit_norm_and_deterministic():
    e = HashingEmbedder()
    a, b = e.embed("My order number is A-4417"), e.embed("My order number is A-4417")
    assert np.allclose(a, b) and abs(np.linalg.norm(a) - 1) < 1e-5


def test_embed_similarity_orders_sensibly():
    e = HashingEmbedder()
    q = e.embed("what is the order number")
    assert q @ e.embed("my order number is B-9032") > q @ e.embed("the weather is awful")


def test_empty_text_zero_vector():
    assert not HashingEmbedder().embed("the a an").any()


def ev(i, cid="c1", text="hello", ts=None):
    return Event(f"e{i}", cid, text, float(ts if ts is not None else i))


def test_store_add_search_isolated_per_customer():
    s = MemoryStore()
    s.add(ev(1, "c1", "I live in Pune now"))
    s.add(ev(2, "c2", "I live in Leeds now"))
    hits = s.search("c1", "which city do they live in", 5)
    assert [h[0].text for h in hits] == ["I live in Pune now"]
    assert s.search("nobody", "x") == []


def test_dedupe_by_event_id():
    s = MemoryStore()
    assert s.add(ev(1)) and not s.add(ev(1))
    assert s.duplicates == 1 and s.count("c1") == 1


def test_growth_and_eviction():
    s = MemoryStore(max_per_customer=40)
    for i in range(100):
        s.add(ev(i, text=f"message number {i}"))
    assert s.count("c1") == 40
    assert s.recent("c1", 1)[0].event_id == "e99"
    assert all(int(h[0].event_id[1:]) >= 60 for h in s.search("c1", "message", 40))


def test_recency_weight_prefers_newer():
    s = MemoryStore(recency_tau=100)
    s.add(ev(1, text="I live in Pune now", ts=0))
    s.add(ev(2, text="I live in Leeds now", ts=1000))
    assert s.search("c1", "where do I live", 1, recency_weight=1.0)[0][0].text.endswith("Leeds now")


def test_search_fresh_puts_newest_near_tie_first():
    s = MemoryStore()
    s.add(ev(1, text="My order number is A-4417.", ts=0))
    for i in range(2, 10):
        s.add(ev(i, text="Thanks, that helps.", ts=i))
    s.add(ev(10, text="My order number is B-9032.", ts=10))
    assert "B-9032" in s.search_fresh("c1", "What is the order number?", 3)[0][0].text


@pytest.mark.parametrize("bad", [{}, {"event_id": 1, "customer_id": 2, "ts": 1}, {"event_id": 1, "customer_id": 2, "ts": 1, "text": " "}])
def test_event_validation(bad):
    with pytest.raises(ValueError):
        Event.from_dict(bad)


def test_percentile():
    assert percentile(list(range(101)), 95) == 95
    assert np.isnan(percentile([], 50))
