"""One subject on two benchmarks (linked through theta), optionally with one
other subject on each: hier vs exact by quadrature (theta x mu_b x delta_b)."""
import math, sys
import numpy as np
from common import HierPredictor, Hyper, item, subject, sig
from exact1 import htab


def exact_link(h, T1, T2, O1, O2):
    S = h.sigma_d ** 2 + h.sigma_g ** 2
    H = htab(S, 0.0)
    vth = h.sigma_theta ** 2 + h.sigma_attr ** 2
    V = vth + h.sigma_delta ** 2
    th = np.linspace(-9 * math.sqrt(vth), 9 * math.sqrt(vth), 121)
    lth = -0.5 * th ** 2 / vth
    if h.nu_mu > 0:
        mu = np.linspace(h.mu0 - 40, h.mu0 + 40, 401)
        lmu = -0.5 * (h.nu_mu + 1) * np.log1p((mu - h.mu0) ** 2 / (h.nu_mu * h.sigma_mu ** 2))
    else:
        mu = np.linspace(h.mu0 - 9 * h.sigma_mu, h.mu0 + 9 * h.sigma_mu, 241)
        lmu = -0.5 * ((mu - h.mu0) / h.sigma_mu) ** 2
    sd = h.sigma_delta
    d = np.linspace(-9 * sd, 9 * sd, 201)
    ld = -0.5 * d ** 2 / sd ** 2
    u = np.linspace(-10 * math.sqrt(V), 10 * math.sqrt(V), 401)
    lu = -0.5 * u ** 2 / V

    def Zo(O):
        if O is None:
            return np.zeros(len(mu))
        k, n = O
        t = lu[None, :] + H.loglik(mu[:, None] + u[None, :], k, n)
        mx = t.max(1)
        return mx + np.log(np.exp(t - mx[:, None]).sum(1))

    def bench(Tk, O, pred=False):
        k, n = Tk
        E = th[:, None, None] + mu[None, :, None] + d[None, None, :]
        L = (lmu + Zo(O))[None, :, None] + ld[None, None, :] + H.loglik(E, k, n)
        mx = L.max(axis=(1, 2))
        W = np.exp(L - mx[:, None, None])
        Z = mx + np.log(W.sum(axis=(1, 2)))
        if pred:
            num = (W * H.plain(E)).sum(axis=(1, 2)) / W.sum(axis=(1, 2))
            return Z, num
        return Z
    Z1 = bench(T1, O1)
    Z2, num = bench(T2, O2, pred=True)
    L = lth + Z1 + Z2
    w = np.exp(L - L.max())
    return float(w @ num / w.sum())


def labeled(T1, T2, O1, O2):
    s = subject("linked")
    lab = []
    for b, Tk, O in (("bench_1", T1, O1), ("bench_2", T2, O2)):
        k, n = Tk
        lab += [[[s, item(j, b)], int(j < k)] for j in range(n)]
        if O is not None:
            ko, no = O
            lab += [[[subject(f"other {b}"), item(500 + j, b)], int(j < ko)] for j in range(no)]
    return lab, [s, item(999, "bench_2")]


CASES = [((31, 31), (0, 1), None, None), ((0, 31), (1, 1), None, None), ((31, 31), (0, 3), (3, 31), (0, 7)),
         ((0, 15), (2, 3), (15, 15), (7, 7)), ((20, 31), (0, 0), None, (5, 31))]
if __name__ == "__main__":
  for label, h in (("gauss w=0.3", Hyper().relink(0.3)), ("gauss default", Hyper()),
                   ("t3 w=0.3", Hyper(sigma_mu=1.829, nu_mu=3.0).relink(0.3))):
      for T1, T2, O1, O2 in CASES:
          ex = exact_link(h, T1, T2, O1, O2)
          lab, inp = labeled(T1, T2, O1, O2)
          ps = [HierPredictor(None, h, floor=False, slip=False, line=ln).predict(inp, lab) for ln in (True, False)]
          off = HierPredictor(None, h, floor=False, slip=False, link=False).predict(inp, lab)
          print(f"{label:<14} T1 {T1} T2 {T2} O1 {O1} O2 {O2}: exact {ex:.4f} line {ps[0]:.4f} ({ps[0]-ex:+.4f}) "
                f"laplace {ps[1]:.4f} ({ps[1]-ex:+.4f}) link-off {off:.4f}", flush=True)
