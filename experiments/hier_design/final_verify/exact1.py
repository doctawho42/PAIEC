"""Exact posterior predictives on one benchmark by nested quadrature, written
from the model statement alone:

    eta = mu + u_s - e_i,  u_s = theta_s + delta_s ~ N(0, V) (no attribute prior),
    mu ~ N(mu0, sigma_mu^2) or t_nu(mu0, sigma_mu), e_i ~ N(0, S) per item,
    P(y = 1 | eta) = c + (1 - c) sig(eta).

Subjects: others (k successes of n on fresh items, plus labels on one shared
item i*), and the target T (kT of nT on fresh items). Target item: fresh or i*.
Given (mu, e*) the subjects are independent; each u_s is integrated on a grid.
"""
import math

import numpy as np

from common import Hfun, sig

_Z = np.linspace(-12, 12, 2401)
_WZ = np.exp(-_Z ** 2 / 2)
_WZ /= _WZ.sum()
_TAB = {}


def htab(S, c):
    if (S, c) not in _TAB:
        _TAB[(S, c)] = Hfun(S, c)
    return _TAB[(S, c)]


def exact_bench(h, others, target, star_target, c=0.0, nmu=241, nu_=None, ne=161):
    """others: [(k, n, star_labels)], target: (kT, nT). Returns E[sig(eta_T)] for
    a fresh item (star_target False) or the shared item i* (True)."""
    S = h.sigma_d ** 2 + h.sigma_g ** 2
    V = h.sigma_theta ** 2 + h.sigma_attr ** 2 + h.sigma_delta ** 2
    H, Hp = htab(S, c), htab(S, 0.0)
    nu = h.nu_mu if nu_ is None else nu_
    if nu > 0:
        mu = np.linspace(h.mu0 - 40, h.mu0 + 40, 2 * nmu + 1)
        lmu = -0.5 * (nu + 1) * np.log1p((mu - h.mu0) ** 2 / (nu * h.sigma_mu ** 2))
    else:
        mu = np.linspace(h.mu0 - 9 * h.sigma_mu, h.mu0 + 9 * h.sigma_mu, nmu)
        lmu = -0.5 * ((mu - h.mu0) / h.sigma_mu) ** 2
    sdv = math.sqrt(V)
    u = np.linspace(-10 * sdv, 10 * sdv, 401)
    lu = -0.5 * (u / sdv) ** 2
    se = math.sqrt(S)
    anystar = star_target or any(len(st) for _, _, st in others)
    if anystar:
        e = np.linspace(-9 * se, 9 * se, ne)
        le = -0.5 * (e / se) ** 2
    else:
        e, le = np.zeros(1), np.zeros(1)
    # log joint over (mu, e) of the others; target numerator/denominator
    L = lmu[:, None] + le[None, :]
    A = mu[:, None] + u[None, :]                       # (M, U) eta before e
    for k, n, star in others:
        base = lu[None, :] + H.loglik(A, k, n)          # (M, U)
        if len(star):
            ll = np.zeros((len(mu), len(u), len(e)))
            for y in star:
                z = A[:, :, None] - e[None, None, :]
                p = c + (1 - c) * sig(z)
                ll += np.log(np.clip(p if y else 1 - p, 1e-300, 1))
            tot = base[:, :, None] + ll
        else:
            tot = np.repeat(base[:, :, None], len(e), axis=2)
        mx = tot.max(1, keepdims=True)
        L += (mx[:, 0, :] + np.log(np.exp(tot - mx).sum(1)))
    kT, nT = target
    baseT = lu[None, :] + H.loglik(A, kT, nT)            # (M, U)
    mxT = baseT.max(1, keepdims=True)
    wT = np.exp(baseT - mxT)                              # (M, U)
    if star_target:
        f = sig(A[:, :, None] - e[None, None, :])        # (M, U, E)
        num = np.einsum("mu,mue->me", wT, f)
    else:
        num = (wT * Hp.plain(A)).sum(1)[:, None] * np.ones((1, len(e)))
    den = wT.sum(1)[:, None] * np.ones((1, len(e)))
    L = L + mxT
    W = np.exp(L - L.max())
    return float((W * num).sum() / (W * den).sum())
