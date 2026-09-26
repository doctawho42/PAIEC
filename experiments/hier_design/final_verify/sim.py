"""Simulate official-protocol-shaped runs from the hier model itself, and an
HMC reference for the exact posterior predictive (collapsed posterior of x,
item residuals integrated by Problem's quadrature, which earlier rounds checked
against brute force)."""
import math

import numpy as np

from common import HierPredictor, Hyper, item, sig, subject

BUDGETS = (0, 1, 3, 7, 15, 31)
_GZ, _GW = np.polynomial.hermite.hermgauss(24)
_GZ, _GW = math.sqrt(2) * _GZ, _GW / _GW.sum()


def simulate(rng, h, pairs, n_items=160, levels=4, scope="pair", mu=None, feats=True):
    """pairs: list of (subject index, benchmark index). Every pair holds all n_items
    items of its benchmark, split 50/50 (per pair or per benchmark), acquisition
    stream in random order. Returns {B: labeled}, targets [(input, outcome, pair)]."""
    subs = sorted({s for s, _ in pairs})
    bens = sorted({b for _, b in pairs})
    theta = {s: rng.normal(0, math.hypot(h.sigma_theta, h.sigma_attr)) for s in subs}
    world = {}
    for b in bens:
        m = rng.normal(h.mu0, h.sigma_mu) if mu is None else mu
        u = rng.normal(0, h.sigma_g, levels)
        grp = rng.integers(0, levels, n_items)
        e = rng.normal(0, h.sigma_d if feats else math.hypot(h.sigma_d, h.sigma_g), n_items)
        its = [item(j, f"bench_{b}", f"tier=t{grp[j]}" if feats else "") for j in range(n_items)]
        eta_i = -(u[grp] if feats else 0) - e
        split = rng.permutation(n_items)
        world[b] = (m, eta_i, its, split)
    labeled = {B: [] for B in BUDGETS}
    targets = []
    for pi, (s, b) in enumerate(pairs):
        m, eta_i, its, split = world[b]
        d = rng.normal(0, h.sigma_delta)
        y = (rng.random(len(its)) < sig(m + theta[s] + d + eta_i)).astype(int)
        order = split if scope == "benchmark" else rng.permutation(len(its))
        acq, ev = order[: len(its) // 2], order[len(its) // 2:]
        acq = rng.permutation(acq)
        sj = subject(f"model {s}")
        for B in BUDGETS:
            labeled[B] += [[[sj, its[j]], int(y[j])] for j in acq[:B]]
        targets += [([sj, its[j]], int(y[j]), pi) for j in ev]
    return labeled, targets


def hmc(prob, post, n=1200, burn=300, eps=0.35, L=12, seed=0, thin=3):
    """Samples of x from the collapsed posterior, whitened by the Laplace cov."""
    rng = np.random.default_rng(seed)
    Lc = np.linalg.cholesky(post.cov)
    xh, e0 = post.x, post.it.mode

    def U(z):
        st = prob.state(xh + Lc @ z, e0)
        return -st.lp, -(Lc.T @ prob.grad(st)), st
    z = np.zeros(len(xh))
    u, gu, st = U(z)
    out, acc = [], 0
    for i in range(n + burn):
        p = rng.standard_normal(len(z))
        z1, p1 = z.copy(), p - 0.5 * eps * gu
        for l in range(L):
            z1 = z1 + eps * p1
            u1, g1, st1 = U(z1)
            if l < L - 1:
                p1 = p1 - eps * g1
        p1 = p1 - 0.5 * eps * g1
        dH = (u + 0.5 * p @ p) - (u1 + 0.5 * p1 @ p1)
        if math.isfinite(dH) and math.log(rng.random()) < dH:
            z, u, gu, st = z1, u1, g1, st1
            acc += i >= burn
        if i >= burn and (i - burn) % thin == 0:
            out.append(st)
    return out, acc / n


def mcmc_predict(fit, states, targets):
    """Posterior predictive E[sig(eta)] per target, averaging over HMC states:
    a'x + m, the untouched Gaussian parts (v) by Gauss-Hermite, the own labeled
    item's (or single folded carrier's) residual by its nodes given x."""
    tgs = [fit._target(s, it) for (s, it), _, _ in targets]
    out = np.zeros(len(tgs))
    for st in states:
        x = st.x
        for k, tg in enumerate(tgs):
            mean = tg.m + sum(c * x[cc] for cc, c in tg.a.items())
            v = tg.v
            if tg.j is not None:
                Q, E = st.it.nodes(tg.j)
                sh = -E
                q = Q
            elif tg.fold:
                assert len(tg.fold) == 1
                (jj, su), = tg.fold.items()
                kk = su / float(fit.prob.s2e[jj])
                Q, E = st.it.nodes(jj)
                sh, q, v = -kk * E, Q, v + su * (1 - kk)
            else:
                sh, q = np.zeros(1), np.ones(1)
            z = mean + sh[:, None] + math.sqrt(max(v, 0)) * _GZ[None, :]
            out[k] += float(q @ (sig(z) @ _GW))
    return out / len(states)
