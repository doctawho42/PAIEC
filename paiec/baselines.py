"""Reference predictors, including a faithful copy of the official empirical mean.

const, empirical_mean, smoothed_mean and pooled_anchor are pure functions of
(input, labeled) in the official format: subjects match on their full dict,
benchmarks on the item's anonymous benchmark_id. irt_predictor needs the private
item keys of the legacy evaluator.
"""
import numpy as np

sig = lambda x: 1 / (1 + np.exp(-x))


def const(v=0.5):
    def f(input, labeled=None):
        return v
    return f


def empirical_mean(input, labeled=None):
    """The organisers' baseline (third_party/paiec_baseline/empirical_mean/model.py):
    successes / observations over labels whose subject dict equals the target's
    and whose benchmark_id matches, 0.5 without any. Unsmoothed, so one label
    drives it to 0 or 1. Raises on the same malformed evidence as the original."""
    subject, item = input
    if not labeled:
        return 0.5
    bm = item.get("benchmark_id")
    if not isinstance(bm, str) or not bm:
        raise ValueError("The mean predictor requires an anonymous item benchmark_id.")
    s = n = 0
    for (osub, oitem), y in labeled:
        if osub != subject:
            continue
        obm = oitem.get("benchmark_id")
        if not isinstance(obm, str) or not obm:
            raise ValueError("Acquired items must include an anonymous benchmark_id.")
        if obm != bm:
            continue
        if y not in (0, 1):
            raise ValueError("Acquired responses must be binary (0 or 1).")
        s += int(y)
        n += 1
    return float(s / n) if n else 0.5


def smoothed_mean(prior_n=4.0, prior_p=0.5):
    """The same thing with a Beta prior, Beta(2, 2) at the defaults. The cheapest
    fix to the baseline's pathology."""
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


def pooled_anchor(tau=4.0, tau2=4.0, tau3=4.0):
    """Own labels shrunk toward the benchmark's mean over every subject's labels.

        p   = (k_sb + tau  * m_b) / (n_sb + tau)
        m_b = (k_b  + tau2 * m_s) / (n_b  + tau2)    all subjects, this benchmark_id
        m_s = (k_s  + tau3 * 0.5) / (n_s  + tau3)    this subject, any benchmark

    The official protocol hands every target the labels of every pair in the
    run, so a pair with one label of its own still sees dozens on its benchmark
    and some on its subject elsewhere. The three strengths are untuned.
    """
    def f(input, labeled=None):
        subject, item = input
        bm = item.get("benchmark_id")
        ksb = nsb = kb = nb = ks = ns = 0
        for (osub, oitem), y in (labeled or []):
            y = int(y)
            same_s, same_b = osub == subject, oitem.get("benchmark_id") == bm
            if same_b:
                kb += y; nb += 1
            if same_s:
                ks += y; ns += 1
                if same_b:
                    ksb += y; nsb += 1
        ms = (ks + tau3 * 0.5) / (ns + tau3)
        mb = (kb + tau2 * ms) / (nb + tau2)
        return float((ksb + tau * mb) / (nsb + tau))
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
