"""Reference predictors, including a faithful copy of the official empirical mean."""
import numpy as np

sig = lambda x: 1 / (1 + np.exp(-x))


def const(v=0.5):
    def f(input, labeled=None):
        return v
    return f


def empirical_mean(input, labeled=None):
    """Official baseline: successes / observations for this subject-benchmark pair."""
    subject, item = input
    if not labeled:
        return 0.5
    bm = item.get("benchmark_id")
    s = n = 0
    for (osub, oitem), y in labeled:
        if osub != subject or oitem.get("benchmark_id") != bm:
            continue
        s += int(y)
        n += 1
    return float(s / n) if n else 0.5


def smoothed_mean(prior_n=4.0, prior_p=0.5):
    """The same thing with a Beta prior. The cheapest fix to the baseline's pathology."""
    def f(input, labeled=None):
        subject, item = input
        bm = item.get("benchmark_id")
        s = n = 0
        for (osub, oitem), y in (labeled or []):
            if osub != subject or oitem.get("benchmark_id") != bm:
                continue
            s += int(y); n += 1
        return float((s + prior_n * prior_p) / (n + prior_n))
    return f


# --- 1-D IRT with a Laplace posterior on ability ---------------------------

def _lap(a, b, tau2, c, d, y, pm=0.0, pv=1.0, it=12):
    m = pm
    for _ in range(it):
        k = 1 / np.sqrt(1 + (np.pi / 8) * tau2 * a**2)
        s = sig(k * a * (m - b))
        p = np.clip(c + (d - c) * s, 1e-6, 1 - 1e-6)
        dp = (d - c) * s * (1 - s) * k * a
        g = np.sum((y - p) * dp / (p * (1 - p))) - (m - pm) / pv
        h = np.sum(dp * dp / (p * (1 - p))) + 1 / pv
        m += g / max(h, 1e-9)
    k = 1 / np.sqrt(1 + (np.pi / 8) * tau2 * a**2)
    s = sig(k * a * (m - b))
    p = np.clip(c + (d - c) * s, 1e-6, 1 - 1e-6)
    dp = (d - c) * s * (1 - s) * k * a
    return m, 1 / (np.sum(dp * dp / (p * (1 - p))) + 1 / pv)


def irt_predictor(item_params, tau2=0.6, prior_v=1.0):
    """item_params: item_key -> (a, b, c, d). Cached per labelled-set identity."""
    cache = {}

    def key_of(item):
        return item.get("_key") or item["item_content"]

    def f(input, labeled=None):
        subject, item = input
        bm = item.get("benchmark_id")
        rows = [(key_of(oi), int(y)) for (os_, oi), y in (labeled or [])
                if os_ == subject and oi.get("benchmark_id") == bm]
        ck = (id(subject), bm, tuple(rows))
        if ck not in cache:
            if rows:
                P = np.array([item_params.get(k, (1.0, 0.0, 0.0, 0.98)) for k, _ in rows])
                y = np.array([v for _, v in rows], float)
                cache[ck] = _lap(P[:, 0], P[:, 1], tau2, P[:, 2], P[:, 3], y, 0.0, prior_v)
            else:
                cache[ck] = (0.0, prior_v)
        m, v = cache[ck]
        a, b, c, d = item_params.get(key_of(item), (1.0, 0.0, 0.0, 0.98))
        k = 1 / np.sqrt(1 + (np.pi / 8) * (v + tau2) * a**2)
        return float(c + (d - c) * sig(k * a * (m - b)))
    return f
