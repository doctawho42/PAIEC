"""Analytic ceilings per pair, computed directly rather than through the evaluator.

  const_half   : Brier of always predicting 0.5           -> exactly 0.25
  pair_mean    : Brier of the pair's true accuracy        -> mu*(1-mu) in expectation
  item_loo     : Brier of the leave-one-out item mean     -> honest estimate of E[p(1-p)]
  unbiased_pq  : k(n-k)/(n(n-1)) averaged over items with n>=2, the textbook
                 unbiased estimator of p(1-p); no fitting, so no optimism.
"""
import numpy as np
from collections import defaultdict
from paiec import evaluator as E


def pair_floors(pair):
    _, ev_keys = E.split_pair(pair)
    ev = set(ev_keys)
    by = defaultdict(list)
    for r in pair.responses:
        by[r.item_key].append(r.label)
    all_lab = np.array([r.label for r in pair.responses], float)
    mu = all_lab.mean()
    se_half = se_mu = se_loo = 0.0
    n_ev = 0
    pq, n_pq = 0.0, 0
    for k, labs in by.items():
        a = np.array(labs, float)
        n, s = len(a), a.sum()
        if n >= 2:
            pq += s * (n - s) / (n * (n - 1)); n_pq += 1
        if k not in ev:
            continue
        loo = (s - a) / (n - 1) if n >= 2 else np.full(n, mu)
        se_half += float(((0.5 - a) ** 2).sum())
        se_mu += float(((mu - a) ** 2).sum())
        se_loo += float(((loo - a) ** 2).sum())
        n_ev += n
    return dict(const_half=se_half / n_ev, pair_mean=se_mu / n_ev,
                item_loo=se_loo / n_ev, unbiased_pq=pq / n_pq if n_pq else np.nan,
                mu=mu, n_ev=n_ev, frac_rep=float(np.mean([len(v) > 1 for v in by.values()])))
