"""Calibration and Brier on runs simulated from the model (truth: TRUE hyper),
predicted with several hyperparameter sets, official budgets, ALC weights.
Usage: c1_calib.py formative|dense runs jobs"""
import math, sys, time, json
import numpy as np
from multiprocessing import Pool
from common import HierPredictor, Hyper
from paiec import prior as PR
from sim import simulate, BUDGETS

W = dict(zip(BUDGETS, (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)))
TRUE = Hyper()
REF = dict(PR.REFERENCE)
HYP = {
    "true": TRUE,
    "true, line off": TRUE,
    "true, t3 mixture": Hyper(sigma_mu=1.697, nu_mu=3.0),
    "fallback mu0=-4.14": Hyper(**{**TRUE.to_dict(), **REF, "mu0": -4.144, "sigma_d": 2.838, "sigma_g": 1.227}),
    "fallback mu0=+0.66": Hyper(**{**TRUE.to_dict(), **REF, "mu0": 0.657, "sigma_d": 3.306, "sigma_g": 1.909}),
    "all REFERENCE": Hyper(**{**TRUE.to_dict(), **REF}),
    "true, sigma_delta 1.0": Hyper(**{**TRUE.to_dict(), "sigma_delta": 1.0}),
}
import os
if os.environ.get("HYP"):
    HYP = {k: v for k, v in HYP.items() if k in os.environ["HYP"].split(",")}
kind = sys.argv[1]
runs = int(sys.argv[2])
jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 6


def one(r):
    rng = np.random.default_rng(1000 + r)
    if kind == "formative":
        nb = 4
        pairs = [(s, s % nb) for s in range(7)] + [(0, 1)]      # subject 0 on two benchmarks
        lab, targets = simulate(rng, TRUE, pairs, n_items=80)
    else:
        lab, targets = simulate(rng, TRUE, [(s, 0) for s in range(26)], n_items=160)
    ys = np.array([y for _, y, _ in targets])
    pid = np.array([p for _, _, p in targets])
    out = {}
    for name, h in HYP.items():
        flags = dict(line=False) if "line off" in name else {}
        rows = {}
        for B in BUDGETS:
            m = HierPredictor(None, h, **flags)
            p = np.array([m.predict(inp, lab[B]) for inp, _, _ in targets])
            # per pair Brier, averaged over pairs
            rows[B] = (p.tolist(), float(np.mean([np.mean((p[pid == k] - ys[pid == k]) ** 2) for k in np.unique(pid)])),
                       m.failures, m.unconverged)
        out[name] = rows
    return ys.tolist(), out


if __name__ == "__main__":
    t = time.time()
    with Pool(jobs) as pool:
        res = pool.map(one, range(runs))
    print(f"{kind}: {runs} runs in {time.time()-t:.0f}s")
    alc = {n: np.array([sum(W[B] * o[n][B][1] for B in BUDGETS) for _, o in res]) for n in HYP}
    for n in HYP:
        d = alc[n] - alc["true"]
        per_b = {B: np.mean([o[n][B][1] for _, o in res]) for B in BUDGETS}
        fails = sum(o[n][B][2] for _, o in res for B in BUDGETS)
        unconv = sum(o[n][B][3] for _, o in res for B in BUDGETS)
        print(f"{n:<22} ALC {alc[n].mean():.5f}  minus true {d.mean():+.5f} +- {d.std(ddof=1)/math.sqrt(runs):.5f}  "
              + " ".join(f"B{B}:{v:.4f}" for B, v in per_b.items()) + f"  failures {fails} unconverged {unconv}")
    # calibration of the true-hyper predictions, all budgets pooled and per budget
    for n in [x for x in ("true", "true, t3 mixture", "all REFERENCE") if x in HYP]:
        P = np.concatenate([np.array(o[n][B][0]) for ys, o in res for B in BUDGETS])
        Y = np.concatenate([np.array(ys) for ys, o in res for B in BUDGETS])
        bins = np.minimum((P * 10).astype(int), 9)
        tab = []
        for b in range(10):
            s = bins == b
            if s.sum():
                tab.append(f"{b/10:.1f}:{P[s].mean():.3f}/{Y[s].mean():.3f}({s.sum()})")
        print(f"calibration [{n}] pred/obs(n): " + " ".join(tab))
        print(f"   mean pred {P.mean():.4f} mean obs {Y.mean():.4f}")
        for B in BUDGETS:
            Pb = np.concatenate([np.array(o[n][B][0]) for ys, o in res])
            Yb = np.concatenate([np.array(ys) for ys, o in res])
            bb = np.minimum((Pb * 5).astype(int), 4)
            gaps = [abs(Pb[bb == k].mean() - Yb[bb == k].mean()) for k in range(5) if (bb == k).sum() > 200]
            ece = sum(abs(Pb[bb == k].mean() - Yb[bb == k].mean()) * (bb == k).sum() for k in range(5) if (bb == k).sum()) / len(Pb)
            print(f"   B{B}: ECE(5 bins) {ece:.4f} max gap {max(gaps):.4f}")
