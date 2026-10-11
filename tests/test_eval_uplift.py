from cms.memeval import generate, run
from cms.uplift import evaluate


def test_generate_deterministic_and_has_updates():
    e1, c1 = generate(10, seed=3)
    e2, c2 = generate(10, seed=3)
    assert [e.text for e in e1] == [e.text for e in e2]
    assert any(c.outdated for c in c1) and len(c1) == 40


def test_retrieval_beats_window_baseline():
    r = run(n_customers=40)
    assert r["cosine retrieval"]["hit@3"] > r["last-5-turns (no retrieval)"]["hit@3"] + 0.2
    rw = [k for k in r if k.startswith("cosine + recency")][0]
    assert r[rw]["stale@1_on_updated_facts"] <= r["cosine retrieval"]["stale@1_on_updated_facts"]


def test_uplift_recovers_signal():
    r = evaluate(n=6000)
    assert r["corr_pred_true_uplift"] > 0.3
    assert r["observed_uplift_top20pct"] > r["observed_uplift_all"]
