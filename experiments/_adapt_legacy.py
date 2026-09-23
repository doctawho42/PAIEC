"""Per-pair adapter: fit a low-dimensional text->difficulty map on the acquired
labels only. Nothing is learned from the held-out benchmark in advance, so this
is honest under leave-one-benchmark-out.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from paiec import items as D
import pandas as pd

sig = lambda x: 1 / (1 + np.exp(-x))


def build_features(pairs):
    df = D.item_table(pairs)
    X = D.numeric_features(df)
    X = StandardScaler().fit_transform(X)
    return {(b, k): X[i] for i, (b, k) in enumerate(zip(df.benchmark_id, df.item_key))}, X.shape[1]


def make_adapter(feat, dim, ridge_w=8.0, ridge_a=1.0, base_logit=-0.4, iters=30):
    cache = {}

    def fit(rows):
        if not rows:
            return base_logit, np.zeros(dim)
        X = np.array([feat[r[0]] for r in rows])
        y = np.array([r[1] for r in rows], float)
        a, w = base_logit, np.zeros(dim)
        for _ in range(iters):
            p = np.clip(sig(a + X @ w), 1e-6, 1 - 1e-6)
            s = p * (1 - p)
            g = np.concatenate([[np.sum(y - p) - ridge_a * (a - base_logit)],
                                X.T @ (y - p) - ridge_w * w])
            Z = np.column_stack([np.ones(len(y)), X])
            H = Z.T @ (Z * s[:, None]) + np.diag([ridge_a] + [ridge_w] * dim)
            step = np.linalg.solve(H + 1e-8 * np.eye(dim + 1), g)
            a += step[0]; w += step[1:]
            if np.max(np.abs(step)) < 1e-8:
                break
        return float(a), w

    def predict(input, labeled=None):
        subject, item = input
        sid, bid = subject.get("_sid"), item.get("benchmark_id")
        rows = tuple(sorted(((oi["benchmark_id"], oi["_key"]), int(y))
                            for (os_, oi), y in (labeled or [])
                            if os_.get("_sid") == sid and oi.get("benchmark_id") == bid))
        ck = (sid, bid, rows)
        if ck not in cache:
            cache[ck] = fit([r for r in rows if r[0] in feat])
        a, w = cache[ck]
        f = feat.get((bid, item["_key"]))
        z = a if f is None else a + float(f @ w)
        return float(np.clip(sig(z), 1e-4, 1 - 1e-4))
    return predict
