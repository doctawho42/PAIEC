"""Acquisition that targets the evaluation pool, not its own uncertainty.

Each candidate is scored by how much revealing it would shrink the predictive
variance over the items we expect to be scored on, under the current posterior
over (a, b). The decision is a threshold on that value: take it if it is in the
top k/n of what we have been seeing, because taking the best k of n online is
what a threshold at the k/n quantile does.

The ALC weights make the first label worth 0.9 and labels 16-31 worth 0.1 each,
so the policy is deliberately pickier early, when the label still counts toward
every budget, and looser later. gamma controls how much.
"""
import numpy as np
from paiec import evaluator as E
from paiec.fitting import fit_ab, sig

WMARG = {}
for k in range(1, 32):
    WMARG[k] = sum(w for w, b in zip(E.WEIGHTS, E.BUDGETS) if b >= k)


def stream(order_z, order_y, m_a, v_a=2.0, v_b=0.25, max_labels=31,
           gamma=0.5, warmup=12, slip=0.02):
    """Walk the candidates once and return the indices taken, in order."""
    n_all = len(order_z)
    taken, seen_vals = [], []
    Zt, Yt = [], []
    a, b = m_a, 1.0
    C = np.array([[v_a, 0.0], [0.0, v_b]])
    for pos in range(n_all):
        k_left = max_labels - len(taken)
        n_left = n_all - pos
        if k_left <= 0:
            break
        if k_left >= n_left:
            taken.append(pos); Zt.append(order_z[pos]); Yt.append(order_y[pos])
            a, b, va, vb = fit_ab(np.array(Zt), np.array(Yt, float), m_a, v_a, 1.0, v_b)
            C = np.array([[va, 0.0], [0.0, vb]])
            continue
        proxy = order_z[max(0, pos - 200):pos + 1]
        z = order_z[pos]
        u = np.array([1.0, z])
        p = float(np.clip(sig(a + b * z), 1e-6, 1 - 1e-6))
        w = p * (1 - p)
        Cu = C @ u
        denom = 1.0 + w * float(u @ Cu)
        dC = np.outer(Cu, Cu) * (w / denom)          # C - C' after this label
        pp = np.clip(sig(a + b * proxy), 1e-6, 1 - 1e-6)
        g = pp * (1 - pp)
        U = np.column_stack([np.ones(len(proxy)), proxy])
        val = float(np.sum(g * g * np.einsum("ij,jk,ik->i", U, dC, U)))
        seen_vals.append(val)
        if len(seen_vals) < warmup:
            take = True
        else:
            f = (0.9 / WMARG[min(len(taken) + 1, 31)]) ** gamma
            q = 1.0 - min(0.999, (k_left / n_left) * f)
            take = val >= np.quantile(seen_vals[-300:], max(0.0, min(0.999, q)))
        if take:
            taken.append(pos); Zt.append(z); Yt.append(order_y[pos])
            a, b, va, vb = fit_ab(np.array(Zt), np.array(Yt, float), m_a, v_a, 1.0, v_b)
            C = np.array([[va, 0.0], [0.0, vb]])
    return taken
