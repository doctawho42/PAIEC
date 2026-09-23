"""How much of the stateful predictor's score comes from the evaluation order?

paiec.evaluator scores a pair at all six budgets before moving to the next pair.
A predictor that keeps state therefore sees earlier pairs' full 31-label
trajectories while it is still being scored at budget 0. Whether the real
evaluator interleaves that way or sweeps budget by budget is not documented, and
it matters: budget-major order leaves budget 0 with no labels anywhere.

Run: python experiments/order_sensitivity.py
"""
import time

import numpy as np

from paiec import evaluator as E
from paiec.data import load_pairs
from paiec.predict import Predictor, fit_prior


def budget_major(pairs, make_pred, seed=0):
    """Score every pair at budget 0, then every pair at budget 1, and so on."""
    traces = {}
    pred = make_pred()
    for p in pairs:
        traces[id(p)] = E.acquire(p, pred, seed=seed)
    out = {}
    for B in E.BUDGETS:
        pred = make_pred()                      # state cannot run ahead of the budget
        se_all = []
        for p in pairs:
            labeled = traces[id(p)].acquired[:B]
            _, ev = E.split_pair(p, seed)
            evs = set(ev)
            se = n = 0.0
            for r in p.responses:
                if r.item_key not in evs:
                    continue
                se += (pred([p.subject, r.item], labeled) - r.label) ** 2
                n += 1
            se_all.append(se / n)
        out[B] = float(np.mean(se_all))
    out["ALC"] = sum(w * out[b] for w, b in zip(E.WEIGHTS, E.BUDGETS))
    return out


def main():
    pairs = load_pairs()
    coef, spec = fit_prior(pairs)
    mk = lambda: Predictor(coef, spec).predict

    t = time.time()
    pair_major, _ = E.run_session(pairs, mk(), seed=0)
    print(f"pair-major  (evaluator as replicated) " + " ".join(
        f"B{b}={pair_major[b]:.4f}" for b in E.BUDGETS)
        + f"  ALC {pair_major['ALC']:.4f}  [{time.time()-t:.0f}s]")

    t = time.time()
    bm = budget_major(pairs, mk, seed=0)
    print(f"budget-major (state cannot look ahead) " + " ".join(
        f"B{b}={bm[b]:.4f}" for b in E.BUDGETS)
        + f"  ALC {bm['ALC']:.4f}  [{time.time()-t:.0f}s]")
    print(f"\ndifference in ALC: {bm['ALC'] - pair_major['ALC']:+.4f}")
    print("The pair-major figure is the optimistic one. Quote the budget-major figure "
          "\nuntil the organisers say which order the evaluator uses.")


if __name__ == "__main__":
    main()
