"""Item difficulty free of the subject mix.

The naive target, the share of subjects who solved an item, is confounded: which
subjects attempted an item is not random. In matharena the 2026 contests were only
ever run against 2026-era models, so their items look easy for reasons that have
nothing to do with the items. A Rasch fit on the full observed matrix separates
subject ability from item difficulty and gives a target the text model can honestly
be asked to predict.
"""
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

sig = lambda x: 1 / (1 + np.exp(-x))


def rasch(si, ji, y, n_s, n_i, lam_t=1.0, lam_b=1.0):
    def obj(p):
        th, b = p[:n_s], p[n_s:]
        pr = sig(th[si] - b[ji])
        nll = -np.sum(y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12))
        nll += 0.5 * (lam_t * th @ th + lam_b * b @ b)
        r = y - pr
        return nll, np.concatenate([np.bincount(si, weights=-r, minlength=n_s) + lam_t * th,
                                    np.bincount(ji, weights=r, minlength=n_i) + lam_b * b])
    res = minimize(obj, np.zeros(n_s + n_i), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=800, maxfun=1200))
    return res.x[:n_s], res.x[n_s:]


def item_difficulty(pairs):
    """benchmark -> {item_key: b}, from a Rasch fit on every observed response."""
    out = {}
    by_b = defaultdict(list)
    for p in pairs:
        by_b[p.benchmark_id].append(p)
    for bid, ps in by_b.items():
        keys, kidx = [], {}
        for p in ps:
            for r in p.responses:
                if r.item_key not in kidx:
                    kidx[r.item_key] = len(keys); keys.append(r.item_key)
        sidx = {(p.subject_id, p.benchmark_id): i for i, p in enumerate(ps)}
        si, ji, y = [], [], []
        for p in ps:
            for r in p.responses:
                si.append(sidx[(p.subject_id, p.benchmark_id)])
                ji.append(kidx[r.item_key]); y.append(r.label)
        th, b = rasch(np.array(si), np.array(ji), np.array(y, float), len(ps), len(keys))
        out[bid] = dict(zip(keys, b))
    return out
