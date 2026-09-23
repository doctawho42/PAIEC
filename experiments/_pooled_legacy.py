"""Per-benchmark text model fitted at test time on pooled acquired labels.

At budget B every pair has revealed B labels. Pool them by benchmark, fit a text
model on that pool, and use it to score every item of the benchmark including
ones nobody labelled. Then fit two parameters per pair on that pair's own labels
to place the subject against the benchmark's difficulty scale.

Legitimate only if the evaluator's `labeled` pools across subjects, or if a
submission may keep state between calls. Both are open questions.
"""
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from paiec import evaluator as E

sig = lambda x: 1 / (1 + np.exp(-x))


def fit_ab(Z, Y, prior_b=1.0, ridge_a=1.0, ridge_b=4.0, iters=40):
    a, b = 0.0, prior_b
    if len(Z) == 0:
        return a, b
    for _ in range(iters):
        p = np.clip(sig(a + b * Z), 1e-6, 1 - 1e-6); w = p * (1 - p)
        g = np.array([np.sum(Y - p) - a * ridge_a,
                      np.sum((Y - p) * Z) - (b - prior_b) * ridge_b])
        H = np.array([[np.sum(w) + ridge_a, np.sum(w * Z)],
                      [np.sum(w * Z), np.sum(w * Z * Z) + ridge_b]])
        s = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        a += s[0]; b += s[1]
        if np.max(np.abs(s)) < 1e-9:
            break
    return float(a), float(b)


def evaluate(pairs, traj, C=0.3, seed=0, use_text=True):
    texts, items_of = {}, defaultdict(list)
    for p in pairs:
        for r in p.responses:
            if (p.benchmark_id, r.item_key) not in texts:
                texts[(p.benchmark_id, r.item_key)] = r.item["item_content"] or ""
                items_of[p.benchmark_id].append(r.item_key)
    out = {}
    for B in E.BUDGETS:
        zmap = {}
        for bid, keys in items_of.items():
            X, Y = [], []
            for p in pairs:
                if p.benchmark_id != bid:
                    continue
                for k, y in traj[(p.subject_id, p.benchmark_id)][:B]:
                    X.append(texts[(bid, k)]); Y.append(y)
            if len(set(Y)) < 2 or not use_text:
                zmap[bid] = {k: 0.0 for k in keys}
                continue
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                strip_accents="unicode")
            Xt = v.fit_transform(X)
            m = LogisticRegression(C=C, max_iter=500).fit(Xt, Y)
            d = m.decision_function(v.transform([texts[(bid, k)] for k in keys]))
            d = (d - d.mean()) / (d.std() + 1e-9)
            zmap[bid] = dict(zip(keys, d))
        se_all = []
        for p in pairs:
            z = zmap[p.benchmark_id]
            rows = traj[(p.subject_id, p.benchmark_id)][:B]
            a, b = fit_ab(np.array([z.get(k, 0.0) for k, _ in rows]),
                          np.array([y for _, y in rows], float)) if rows else (0.0, 1.0)
            _, ev = E.split_pair(p, seed); evs = set(ev)
            se = n = 0.0
            for r in p.responses:
                if r.item_key not in evs:
                    continue
                pr = np.clip(sig(a + b * z.get(r.item_key, 0.0)), 1e-4, 1 - 1e-4)
                se += (pr - r.label) ** 2; n += 1
            se_all.append(se / n)
        out[B] = float(np.mean(se_all))
    return out


def evaluate_blend(pairs, traj, C=0.3, kappa=2.0, seed=0):
    """Direct pooled labels where they exist, text model where they do not."""
    texts, items_of = {}, defaultdict(list)
    for p in pairs:
        for r in p.responses:
            if (p.benchmark_id, r.item_key) not in texts:
                texts[(p.benchmark_id, r.item_key)] = r.item["item_content"] or ""
                items_of[p.benchmark_id].append(r.item_key)
    out = {}
    for B in E.BUDGETS:
        counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for p in pairs:
            for k, y in traj[(p.subject_id, p.benchmark_id)][:B]:
                c = counts[p.benchmark_id][k]; c[0] += 1; c[1] += y
        ztext = {}
        for bid, keys in items_of.items():
            X = [texts[(bid, k)] for k, c in counts[bid].items() for _ in range(c[0])]
            Y = [1] * 0
            X, Y = [], []
            for p in pairs:
                if p.benchmark_id != bid:
                    continue
                for k, y in traj[(p.subject_id, p.benchmark_id)][:B]:
                    X.append(texts[(bid, k)]); Y.append(y)
            if len(set(Y)) < 2:
                ztext[bid] = {k: 0.0 for k in keys}; continue
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                strip_accents="unicode")
            m = LogisticRegression(C=C, max_iter=500).fit(v.fit_transform(X), Y)
            d = m.decision_function(v.transform([texts[(bid, k)] for k in keys]))
            d = (d - d.mean()) / (d.std() + 1e-9)
            ztext[bid] = dict(zip(keys, d))
        se_all = []
        for p in pairs:
            own = defaultdict(lambda: [0, 0])
            for k, y in traj[(p.subject_id, p.benchmark_id)][:B]:
                own[k][0] += 1; own[k][1] += y
            cb = counts[p.benchmark_id]
            raw = {}
            for k in items_of[p.benchmark_id]:
                n, s = cb.get(k, [0, 0]); o = own.get(k, [0, 0])
                n, s = n - o[0], s - o[1]
                raw[k] = (n, np.log((s + 0.5) / (n - s + 0.5)) if n > 0 else 0.0)
            vals = [v for n, v in raw.values() if n > 0]
            mu, sd = (np.mean(vals), np.std(vals) + 1e-9) if vals else (0.0, 1.0)
            zt = ztext[p.benchmark_id]

            def z(k):
                n, v = raw.get(k, (0, 0.0))
                w = n / (n + kappa)
                return w * ((v - mu) / sd) + (1 - w) * zt.get(k, 0.0)

            rows = traj[(p.subject_id, p.benchmark_id)][:B]
            a, b = fit_ab(np.array([z(k) for k, _ in rows]),
                          np.array([y for _, y in rows], float)) if rows else (0.0, 1.0)
            _, ev = E.split_pair(p, seed); evs = set(ev)
            se = n = 0.0
            for r in p.responses:
                if r.item_key not in evs:
                    continue
                se += (np.clip(sig(a + b * z(r.item_key)), 1e-4, 1 - 1e-4) - r.label) ** 2
                n += 1
            se_all.append(se / n)
        out[B] = float(np.mean(se_all))
    return out
