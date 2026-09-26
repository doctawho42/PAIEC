"""Dense runs from the model: the line with the adaptive LINE_MAX cap and the
second-order rest term (new), the old all-or-nothing cap (LINE_CORE = 0: any
overflow falls back to the Gaussian), uncapped, and Laplace, against the HMC
posterior predictive (final_verify/sim.py) and in Brier. Usage: n_s reps seed"""
import sys, time
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))  # the repo
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "final_verify"))
import numpy as np
from multiprocessing import Pool
import paiec.hier as H
from paiec.hier import HierPredictor, Hyper
from sim import simulate, hmc, mcmc_predict

n_s = int(sys.argv[1]); reps = int(sys.argv[2]); seed = int(sys.argv[3])
VARIANTS = (("core 0.2", 600, 0.2, True), ("top 600", 600, float("inf"), True), ("top 1200", 1200, float("inf"), True), ("uncapped", 10**9, 0.2, True),
            ("laplace", 600, 0.2, False))


def one(rep):
    rng = np.random.default_rng([seed, rep])
    h = Hyper()
    lab, targets = simulate(rng, h, [(s, 0) for s in range(n_s)], n_items=int(sys.argv[4]), feats=sys.argv[5] == "1")
    ys = np.array([y for _, y, _ in targets])
    lines = []
    for B in (31,):
        base = HierPredictor(None, h, floor=False, slip=False)
        fit = base.fit_for(lab[B])
        states, acc = hmc(fit.prob, fit.post, n=900, burn=200, seed=rep * 10 + B, thin=3)
        pm = mcmc_predict(fit, states, targets)
        row = {"B": B, "n_lab": len(lab[B]), "acc": acc, "brier_mcmc": float(np.mean((pm - ys) ** 2))}
        for name, cap, core, line in VARIANTS:
            H.LINE_MAX, H.LINE_CORE = cap, core
            m = HierPredictor(None, h, floor=False, slip=False, line=line)
            t = time.perf_counter()
            p = np.array([m.predict(inp, lab[B]) for inp, _, _ in targets])
            row[name] = (float(np.abs(p - pm).mean()), float(np.abs(p - pm).max()),
                         float(np.mean((p - ys) ** 2) - np.mean((pm - ys) ** 2)), time.perf_counter() - t)
        H.LINE_MAX, H.LINE_CORE = 600, 0.2
        lines.append(row)
    return lines


if __name__ == "__main__":
    with Pool(min(reps, 6)) as pool:
        res = pool.map(one, range(reps))
    for rep, rows in enumerate(res):
        for r in rows:
            print(f"n_s {n_s} rep{rep} B{r['B']} labels {r['n_lab']} acc {r['acc']:.2f} | " + " | ".join(
                f"{n}: |d| {r[n][0]:.4f} max {r[n][1]:.3f} dBrier {r[n][2]:+.5f} {r[n][3]:.1f}s" for n, *_ in VARIANTS))
    for B in (31,):
        rows = [r for rows in res for r in rows if r["B"] == B]
        print(f"B{B} mean over reps: " + " | ".join(
            f"{n}: |d| {np.mean([r[n][0] for r in rows]):.4f} dBrier {np.mean([r[n][2] for r in rows]):+.5f} "
            f"time {np.mean([r[n][3] for r in rows]):.1f}s" for n, *_ in VARIANTS))
