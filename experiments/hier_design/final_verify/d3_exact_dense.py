"""One benchmark, n_s subjects drawn from the model (Gaussian or t level), B labels
each on fresh items, no features: hier (line / laplace) against the exact posterior
predictive on a fresh item. Excess Brier (p - pi)^2 against the true success
probability pi = E_e sig(mu + u - e), so no outcome noise."""
import math, sys, time
import numpy as np
from common import HierPredictor, Hyper, item, subject, sig
from exact1 import exact_bench, htab

n_s = int(sys.argv[1]) if len(sys.argv) > 1 else 26
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
nu = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
hsim = Hyper() if nu == 0 else Hyper(sigma_mu=1.829, nu_mu=nu)
h = hsim
S = h.sigma_d ** 2 + h.sigma_g ** 2
V = h.sigma_theta ** 2 + h.sigma_attr ** 2 + h.sigma_delta ** 2
Hp = htab(S, 0.0)
rng = np.random.default_rng(100 + n_s)
acc = {B: {"line": [], "laplace": [], "exact": [], "tow_line": [], "tow_lap": [], "abs_line": [], "abs_lap": []} for B in (1, 3, 7, 15, 31)}
t0 = time.perf_counter()
for rep in range(reps):
    mu = h.mu0 + h.sigma_mu * (rng.standard_normal() if nu == 0 else rng.standard_t(nu))
    u = rng.normal(0, math.sqrt(V), n_s)
    pi = Hp.plain(mu + u)
    e = rng.normal(0, math.sqrt(S), (n_s, 31))
    Y = (rng.random((n_s, 31)) < sig(mu + u[:, None] - e)).astype(int)
    subs = [subject(f"m{s}") for s in range(n_s)]
    for B in (1, 3, 7, 15, 31):
        lab = [[[subs[s], item(1000 * s + j, "b")], int(Y[s, j])] for s in range(n_s) for j in range(B)]
        ml = HierPredictor(None, h, floor=False, slip=False)
        mg = HierPredictor(None, h, floor=False, slip=False, line=False)
        for s in range(n_s):
            k = int(Y[s, :B].sum())
            others = [(int(Y[o, :B].sum()), B, []) for o in range(n_s) if o != s]
            ex = exact_bench(h, others, (k, B), False)
            pl = ml.predict([subs[s], item(99999, "b")], lab)
            pg = mg.predict([subs[s], item(99999, "b")], lab)
            a = acc[B]
            a["exact"].append((ex - pi[s]) ** 2); a["line"].append((pl - pi[s]) ** 2); a["laplace"].append((pg - pi[s]) ** 2)
            tw = np.sign(0.5 - ex)
            a["tow_line"].append((pl - ex) * tw); a["tow_lap"].append((pg - ex) * tw)
            a["abs_line"].append(abs(pl - ex)); a["abs_lap"].append(abs(pg - ex))
        assert ml.failures == 0 and mg.failures == 0
    print(f"rep {rep} done ({time.perf_counter()-t0:.0f}s)", flush=True)
for B, a in acc.items():
    n = len(a["exact"])
    dl = np.array(a["line"]) - np.array(a["exact"]); dg = np.array(a["laplace"]) - np.array(a["exact"])
    print(f"n_s {n_s} nu {nu} B{B:<2}: excess Brier exact {np.mean(a['exact']):.5f}; line {dl.mean():+.5f}+-{dl.std()/math.sqrt(n):.5f} "
          f"laplace {dg.mean():+.5f}+-{dg.std()/math.sqrt(n):.5f}; |p-exact| line {np.mean(a['abs_line']):.4f} (max {np.max(a['abs_line']):.3f}) "
          f"laplace {np.mean(a['abs_lap']):.4f}; toward 0.5 line {np.mean(a['tow_line']):+.4f} laplace {np.mean(a['tow_lap']):+.4f}", flush=True)
