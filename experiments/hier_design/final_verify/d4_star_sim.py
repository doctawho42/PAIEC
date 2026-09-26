"""Labeled-item targets drawn from the model: m others label the target's item i*
(each also B labels on fresh items), the target T has B labels on fresh items.
hier (line / laplace) vs the exact posterior predictive; excess Brier against
the true probability sig(mu + u_T - e*)."""
import math, sys
import numpy as np
from common import HierPredictor, Hyper, sig
from exact1 import exact_bench
from v2_bench import build

m_oth = int(sys.argv[1]); B = int(sys.argv[2]); reps = int(sys.argv[3]); floor = len(sys.argv) > 4 and sys.argv[4] == "floor"
h = Hyper()
S = h.sigma_d ** 2 + h.sigma_g ** 2
V = h.sigma_theta ** 2 + h.sigma_attr ** 2 + h.sigma_delta ** 2
c = h.guess * 0.25 if floor else 0.0
rng = np.random.default_rng(1000 * m_oth + B)
rows = []
for r in range(reps):
    mu = rng.normal(h.mu0, h.sigma_mu)
    u = rng.normal(0, math.sqrt(V), m_oth + 1)
    es = rng.normal(0, math.sqrt(S))
    def draw(eta, n):
        pe = sig(eta - rng.normal(0, math.sqrt(S), n))
        return int((rng.random(n) < c + (1 - c) * pe).sum())
    others = []
    for o in range(m_oth):
        k = draw(mu + u[o], B)
        ystar = int(rng.random() < c + (1 - c) * sig(mu + u[o] - es))
        others.append((k, B, [ystar]))
    kT = draw(mu + u[-1], B)
    pi = c + (1 - c) * sig(mu + u[-1] - es)
    ex = c + (1 - c) * exact_bench(h, others, (kT, B), True, c=c)
    lab, inp = build(others, (kT, B), True, floor)
    pl = HierPredictor(None, h, slip=False, floor=floor).predict(inp, lab)
    pg = HierPredictor(None, h, slip=False, floor=floor, line=False).predict(inp, lab)
    rows.append((ex, pl, pg, pi))
a = np.array(rows)
ex, pl, pg, pi = a.T
n = len(a)
dl, dg = (pl - pi) ** 2 - (ex - pi) ** 2, (pg - pi) ** 2 - (ex - pi) ** 2
print(f"m={m_oth} B={B} floor={floor} n={n}: |line-exact| {np.abs(pl-ex).mean():.4f} (max {np.abs(pl-ex).max():.3f}, 95% {np.quantile(np.abs(pl-ex), .95):.3f}) "
      f"|laplace-exact| {np.abs(pg-ex).mean():.4f} (max {np.abs(pg-ex).max():.3f}); excess Brier exact {np.mean((ex-pi)**2):.5f} "
      f"line {dl.mean():+.5f}+-{dl.std()/math.sqrt(n):.5f} laplace {dg.mean():+.5f}+-{dg.std()/math.sqrt(n):.5f}", flush=True)
