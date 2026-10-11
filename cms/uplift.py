"""Small uplift module: T-learner on a SYNTHETIC dataset with a known true effect.

Scenario (made up): a proactive "memory-aware" follow-up message is sent to a random
half of customers; outcome = repeat purchase. True uplift depends on features, so we
can measure how well the T-learner recovers it (it is not a real-world result).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


def synth(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    t = rng.integers(0, 2, n)
    base = -1.0 + 0.8 * X[:, 0] - 0.5 * X[:, 1]
    tau_logit = 0.9 * (X[:, 2] > 0) - 0.4 * (X[:, 3] > 1)
    p0 = 1 / (1 + np.exp(-base))
    p1 = 1 / (1 + np.exp(-(base + tau_logit)))
    y = rng.random(n) < np.where(t == 1, p1, p0)
    return X, t, y.astype(int), p1 - p0


class TLearner:
    def __init__(self, seed=0):
        self.m0 = GradientBoostingClassifier(random_state=seed, max_depth=3, n_estimators=150)
        self.m1 = GradientBoostingClassifier(random_state=seed, max_depth=3, n_estimators=150)

    def fit(self, X, t, y):
        self.m0.fit(X[t == 0], y[t == 0])
        self.m1.fit(X[t == 1], y[t == 1])
        return self

    def predict_uplift(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


def evaluate(n=20000, seed=0) -> dict:
    X, t, y, tau = synth(n, seed)
    k = n // 2
    model = TLearner(seed).fit(X[:k], t[:k], y[:k])
    pred = model.predict_uplift(X[k:])
    tt, yt, taut = t[k:], y[k:], tau[k:]
    order = np.argsort(-pred)
    top = order[: len(order) // 5]  # target top 20%
    rand_top = np.arange(len(order))
    uplift_obs = lambda idx: yt[idx][tt[idx] == 1].mean() - yt[idx][tt[idx] == 0].mean()
    return {"n_test": int(len(pred)), "pehe": float(np.sqrt(np.mean((pred - taut) ** 2))),
            "corr_pred_true_uplift": float(np.corrcoef(pred, taut)[0, 1]),
            "true_ate": float(taut.mean()),
            "observed_uplift_top20pct": float(uplift_obs(top)),
            "observed_uplift_all": float(uplift_obs(rand_top)),
            "true_uplift_top20pct": float(taut[top].mean())}
