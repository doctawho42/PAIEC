"""Item difficulty from other subjects, then a two-parameter fit per pair.

z_i is the smoothed logit of how often OTHER subjects got item i right. The
target subject's own responses are removed, so nothing leaks. The predictor is
p = sigmoid(a + b*z_i) with a, b fitted on the acquired labels under a ridge
prior centred at (0, 1): at budget 0 that is exactly the other-subjects' rate.
"""
import numpy as np
from collections import defaultdict

sig = lambda x: 1 / (1 + np.exp(-x))


def build_difficulty(pairs):
    """benchmark -> item_key -> (n, k) over all subjects."""
    tot = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    own = defaultdict(lambda: [0, 0])
    for p in pairs:
        for r in p.responses:
            t = tot[p.benchmark_id][r.item_key]
            t[0] += 1; t[1] += r.label
            o = own[(p.subject_id, p.benchmark_id, r.item_key)]
            o[0] += 1; o[1] += r.label
    return tot, own


def make_cf_predictor(pairs, prior_b=1.0, ridge_a=1.0, ridge_b=4.0, iters=40):
    tot, own = build_difficulty(pairs)
    zc = {}

    def z_of(sid, bid, key):
        ck = (sid, bid, key)
        if ck in zc:
            return zc[ck]
        n, k = tot[bid].get(key, [0, 0])
        on, ok = own.get(ck, [0, 0])
        n, k = n - on, k - ok
        z = float(np.log((k + 0.5) / (n - k + 0.5))) if n > 0 else 0.0
        zc[ck] = z
        return z

    cache = {}

    def fit(rows):
        a, b = 0.0, prior_b
        if not rows:
            return a, b
        Z = np.array([r[0] for r in rows]); Y = np.array([r[1] for r in rows], float)
        for _ in range(iters):
            p = np.clip(sig(a + b * Z), 1e-6, 1 - 1e-6)
            w = p * (1 - p)
            g = np.array([np.sum(Y - p) - a * ridge_a,
                          np.sum((Y - p) * Z) - (b - prior_b) * ridge_b])
            H = np.array([[np.sum(w) + ridge_a, np.sum(w * Z)],
                          [np.sum(w * Z), np.sum(w * Z * Z) + ridge_b]])
            step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
            a += step[0]; b += step[1]
            if np.max(np.abs(step)) < 1e-8:
                break
        return float(a), float(b)

    def predict(input, labeled=None):
        subject, item = input
        sid, bid = subject.get("_sid"), item.get("benchmark_id")
        rows = tuple(sorted((oi["_key"], int(y)) for (os_, oi), y in (labeled or [])
                            if os_.get("_sid") == sid and oi.get("benchmark_id") == bid))
        ck = (sid, bid, rows)
        if ck not in cache:
            cache[ck] = fit([(z_of(sid, bid, k), y) for k, y in rows])
        a, b = cache[ck]
        return float(np.clip(sig(a + b * z_of(sid, bid, item["_key"])), 1e-4, 1 - 1e-4))

    return predict
