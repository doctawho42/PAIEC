# Diagnostic: how much item difficulty is recoverable from other subjects'
# acquired labels alone. Superseded by paiec.pipeline, kept to reproduce the
# number reported in docs/findings.md.
"""Diagnostic: how much item difficulty can be recovered from OTHER SUBJECTS'
acquired labels alone, i.e. with no prior data on the benchmark at all.

This is the realistic version of the collaborative-filtering ceiling. It matters
only if the evaluator's `labeled` really does pool across subjects. Not a
faithful evaluator run; it measures a quantity, it does not score a submission.
"""
import numpy as np
from collections import defaultdict
from paiec import evaluator as E

sig = lambda x: 1 / (1 + np.exp(-x))


def collect_trajectories(pairs, predict, acquisition=None, seed=0):
    traj = {}
    for p in pairs:
        t = E.acquire(p, predict, acquisition, seed=seed)
        traj[(p.subject_id, p.benchmark_id)] = list(zip(t.acquired_keys,
                                                        [y for _, y in t.acquired]))
    return traj


def fit_ab(Z, Y, prior_b=1.0, ridge_a=1.0, ridge_b=4.0, iters=40):
    a, b = 0.0, prior_b
    if len(Z) == 0:
        return a, b
    for _ in range(iters):
        p = np.clip(sig(a + b * Z), 1e-6, 1 - 1e-6)
        w = p * (1 - p)
        g = np.array([np.sum(Y - p) - a * ridge_a,
                      np.sum((Y - p) * Z) - (b - prior_b) * ridge_b])
        H = np.array([[np.sum(w) + ridge_a, np.sum(w * Z)],
                      [np.sum(w * Z), np.sum(w * Z * Z) + ridge_b]])
        s = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        a += s[0]; b += s[1]
        if np.max(np.abs(s)) < 1e-9:
            break
    return float(a), float(b)


def evaluate(pairs, traj, seed=0):
    """Brier per budget when z_i comes only from other subjects' first-B labels."""
    by_bench = defaultdict(list)
    for p in pairs:
        by_bench[p.benchmark_id].append(p)
    out = {B: [] for B in E.BUDGETS}
    for B in E.BUDGETS:
        counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for p in pairs:
            for k, y in traj[(p.subject_id, p.benchmark_id)][:B]:
                c = counts[p.benchmark_id][k]
                c[0] += 1; c[1] += y
        for p in pairs:
            own = defaultdict(lambda: [0, 0])
            for k, y in traj[(p.subject_id, p.benchmark_id)][:B]:
                own[k][0] += 1; own[k][1] += y

            def z(k):
                n, s = counts[p.benchmark_id].get(k, [0, 0])
                o = own.get(k, [0, 0])
                n, s = n - o[0], s - o[1]
                return float(np.log((s + 0.5) / (n - s + 0.5))) if n > 0 else 0.0

            rows = traj[(p.subject_id, p.benchmark_id)][:B]
            a, b = fit_ab(np.array([z(k) for k, _ in rows]),
                          np.array([y for _, y in rows], float)) if rows else (0.0, 1.0)
            _, ev = E.split_pair(p, seed)
            evset = set(ev)
            se = n = 0.0
            for r in p.responses:
                if r.item_key not in evset:
                    continue
                pr = np.clip(sig(a + b * z(r.item_key)), 1e-4, 1 - 1e-4)
                se += (pr - r.label) ** 2; n += 1
            out[B].append(se / n)
    return {B: float(np.mean(v)) for B, v in out.items()}
