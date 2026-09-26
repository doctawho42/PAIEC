"""Prototype: a target on a labeled item j read on a 2-D grid over (a'x, J'x),
the other coordinates at their Gaussian conditional mean given both, the exact
log posterior (items moved by either direction integrated, the rest to second
order) weighting each point, and j's residual read node by node at each point.
Monkeypatches paiec.hier._Fit._components when installed."""
import math
import numpy as np
import paiec.hier as H
from paiec.hier import _Items, _mv, _prior_curvature, _log_prior

ORIG = H._Fit._components
UH = 0.75
STATS = {"plane": 0, "fallback": 0}


def _hit2(fit, m1, m2):
    mv1 = np.maximum.reduceat(m1[fit._order], fit._starts[:-1])
    mv2 = np.maximum.reduceat(m2[fit._order], fit._starts[:-1])
    mv = np.maximum(mv1 / max(m1.max(), 1e-300), mv2 / max(m2.max(), 1e-300))
    cand = np.flatnonzero(mv >= H.LINE_FRAC)
    order = cand[np.argsort(-mv[cand], kind="stable")]
    k = int(np.searchsorted(np.cumsum(fit._count[order]), H.LINE_MAX, side="right"))
    if k < len(order) and mv[order[k]] >= H.LINE_CORE:
        return None
    return np.sort(order[:k])


def plane_components(self, post, tg):
    if tg.j is None or not tg.a or not self.model.cfg.line:
        return ORIG(self, post, tg)
    line = self._line(post, tg.a)
    if line is None:
        return ORIG(self, post, tg)
    prob = self.prob
    J = self._J(post, tg.j)
    cs = sorted(set(tg.a) | set(J))
    av = np.array([tg.a.get(c, 0.0) for c in cs]); jv = np.array([J.get(c, 0.0) for c in cs])
    Sa = _mv(post.cov[:, cs], av); SJ = _mv(post.cov[:, cs], jv)
    s2 = float(_mv(Sa[cs], av)); cJa = float(_mv(SJ[cs], av)); vJ = float(_mv(SJ[cs], jv)) - cJa ** 2 / s2
    if vJ < 1e-6:
        return ORIG(self, post, tg)
    d = Sa / s2
    dJ = (SJ - Sa * cJa / s2) / vJ
    if post.eta is None:
        post.eta = prob.eta(post.x)
    delta = np.einsum("os,os->o", prob.vals, d[prob.cols])
    deltaJ = np.einsum("os,os->o", prob.vals, dJ[prob.cols])
    hit = _hit2(self, np.abs(delta), np.abs(deltaJ))
    if hit is None or tg.j not in set(hit.tolist()):
        STATS["fallback"] += 1
        return ORIG(self, post, tg)
    STATS["plane"] += 1
    labs = self._labels(hit)
    li = np.searchsorted(hit, prob.item[labs])
    y, c, s2e, e0 = prob.y[labs], prob.c[labs], prob.s2e[hit], post.it.mode[hit]
    g = post.it.g
    lin_t = float(_mv(g, delta) - _mv(g[labs], delta[labs]))
    lin_u = float(_mv(g, deltaJ) - _mv(g[labs], deltaJ[labs]))
    it0 = _Items(post.eta[labs], y, li, len(hit), s2e, e0, c)
    ct = it0.curvature(delta[labs]).sum(); cu = it0.curvature(deltaJ[labs]).sum()
    cb = it0.curvature(delta[labs] + deltaJ[labs]).sum()
    ctu = 0.5 * (cb - ct - cu)
    pc = _prior_curvature(post.x, prob.m, post.V, post.nu)
    Rtt = max(1 / s2 - ct - float(pc @ (d * d)), 0.0)
    Ruu = max(1 / vJ - cu - float(pc @ (dJ * dJ)), 0.0)
    Rtu = -ctu - float(pc @ (d * dJ))
    sdt, sdu = math.sqrt(s2), math.sqrt(vJ)
    # t nodes: the line's own, where it has weight
    tz = np.concatenate([z[w > 1e-10] for z, w, _ in line.blocks])
    tz = np.unique(tz)
    t = sdt * tz
    uz = UH * np.arange(-12, 13)
    for _ in range(6):
        u = sdu * uz
        T, U = np.meshgrid(t, u, indexing="ij")
        T, U = T.ravel(), U.ravel()
        G = len(T)
        eta = post.eta[labs][None, :] + delta[labs][None, :] * T[:, None] + deltaJ[labs][None, :] * U[:, None]
        item = (np.arange(G)[:, None] * len(hit) + li[None, :]).ravel()
        it = _Items(eta.ravel(), np.tile(y, G), item, G * len(hit), np.tile(s2e, G), np.tile(e0, G), np.tile(c, G))
        X = post.x[None, :] + T[:, None] * d[None, :] + U[:, None] * dJ[None, :]
        ell = it.logL.reshape(G, len(hit)).sum(1) + lin_t * T + lin_u * U \
            - 0.5 * (Rtt * T * T + 2 * Rtu * T * U + Ruu * U * U) + _log_prior(X, prob.m, post.V, post.nu)
        L = ell.reshape(len(t), len(uz))
        top = L.max()
        edge = max(L[:, 0].max(), L[:, -1].max())
        if edge < top - 16:
            break
        uz = UH * np.arange(-12 - 8 * (_ + 1), 13 + 8 * (_ + 1))
    W = np.exp(ell - ell.max()); W /= W.sum()
    pos = int(np.searchsorted(hit, tg.j))
    Q, E = it.nodes(np.arange(G) * len(hit) + pos)
    Q = Q / Q.sum(1)[:, None]
    m = tg.m + sum(coef * float(post.x[cc]) for cc, coef in tg.a.items())
    WW = W[:, None] * Q
    M = (m + T)[:, None] - E
    keep = WW > 1e-14 * WW.max()
    w = WW[keep]
    return w / w.sum(), M[keep], np.full(len(w), max(tg.v, 0.0))


def install():
    H._Fit._components = plane_components


def uninstall():
    H._Fit._components = ORIG
