"""Pick the quadrature over the scale-mixture weight lambda ~ Gamma(nu/2, nu/2)
for a Student-t level: log-space trapezoid vs generalized Gauss-Laguerre, on the
closed-form Gaussian-likelihood case (level observed as N(m_obs, v_obs))."""
import math
import numpy as np
from scipy import special, integrate

nu, s2 = 3.0, 1.829 ** 2
a = nu / 2

def gamma_logpdf(lam):
    return a * math.log(a) - special.gammaln(a) + (a - 1) * np.log(lam) - a * lam

def quantities(lam, m, v):
    V = s2 / lam
    ev = np.exp(-0.5 * m * m / (V + v)) / np.sqrt(V + v)      # evidence up to const
    post = m * V / (V + v)                                    # posterior mean of level
    return ev, post

def exact(m, v):
    f = lambda u: math.exp(gamma_logpdf(math.exp(u)) + u)
    num = integrate.quad(lambda u: f(u) * quantities(math.exp(u), m, v)[0] * quantities(math.exp(u), m, v)[1], -40, 8, limit=500)[0]
    den = integrate.quad(lambda u: f(u) * quantities(math.exp(u), m, v)[0], -40, 8, limit=500)[0]
    return num / den

def trap(K, lo=-9.0, hi=3.0):
    u = np.linspace(lo, hi, K)
    lw = gamma_logpdf(np.exp(u)) + u
    return np.exp(u), np.exp(lw - lw.max())

def genlag(K):
    x, w = special.roots_genlaguerre(K, a - 1)
    return x / a, w

for name, rule in [("trap12", trap(12)), ("trap16", trap(16)), ("trap20", trap(20)),
                   ("trap16w", trap(16, -11, 3.5)), ("gl8", genlag(8)), ("gl12", genlag(12)), ("gl16", genlag(16))]:
    lam, w = rule
    worst = 0
    for m in (0.0, 1.0, 3.0, 6.0, 10.0, 20.0):
        for v in (0.01, 0.1, 1.0, 4.0, 30.0):
            ev, post = quantities(lam, m, v)
            est = (w * ev * post).sum() / (w * ev).sum()
            worst = max(worst, abs(est - exact(m, v)) / (1 + abs(m)))
    print(name, f"worst rel error of posterior mean {worst:.2e}")
