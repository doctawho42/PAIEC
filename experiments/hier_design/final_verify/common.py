"""Independent exact posterior predictives for small hier problems.

Written from the model statement in paiec/hier.py's docstring, not from its code:
    eta = mu_b + theta_s + delta_sb - g_i - e_i
    P(y=1) = c + (1-c) sigmoid(eta)             (likelihood)
    predict = c + (1-c-slip) E[sigmoid(eta)]      (prediction)
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from paiec.hier import Hyper, HierPredictor  # noqa: E402
from paiec import prior as PR  # noqa: E402
from paiec.subjects import Spec  # noqa: E402


def sig(x):
    return 0.5 * (1 + np.tanh(0.5 * np.asarray(x, float)))


def subject(name, **kw):
    s = dict.fromkeys(("normalized_name", "provider", "release_date", "access_date", "harness",
                       "harness_version", "reasoning_effort", "subject_features_extra"), "")
    s.update(normalized_name=name, **kw)
    return s


def item(j, bid, features="", text=None):
    return {"item_content": text or f"question {j} of {bid}", "item_features": features,
            "interactors": "", "benchmark_id": bid}


def mcq(j, bid):
    return item(j, bid, text=f"Question {j}?\nA) one\nB) two\nC) three\nD) four")


def toy_prior():
    spec = Spec(["openai", "anthropic"], ["high"], 500.0, 3.0)
    coef = np.array([0.1, 0.8, 0.0, 0.2, 0.1, -0.3, 0.3, 0.2, 0.1, 0.4, -0.2, 0.3])
    table = {"gizmo 2": [0.9, 0.6, 2]}
    return PR.SubjectPrior(coef, spec, 0.01 * np.eye(len(coef)), table,
                           {"raw": 2.0, "resid": 2.2}, {"benchmarks": ["x"]})


# E over e ~ N(0, S) by a fine grid, tabulated on eta
_Z = np.linspace(-12, 12, 4801)
_WZ = np.exp(-_Z ** 2 / 2)
_WZ /= _WZ.sum()


class Hfun:
    """h1(eta) = E_e[c + (1-c) sig(eta - e)], e ~ N(0,S): P(y=1 | eta) with item integrated.
    plain(eta) = E_e[sig(eta - e)]."""

    def __init__(self, S, c=0.0, lo=-80, hi=80, n=64001):
        self.t = np.linspace(lo, hi, n)
        sd = math.sqrt(S)
        # chunked to save memory
        vals = np.empty(n)
        for a in range(0, n, 4000):
            vals[a:a + 4000] = sig(self.t[a:a + 4000, None] - sd * _Z[None, :]) @ _WZ
        self.plain_tab = vals
        self.c = c

    def plain(self, eta):
        return np.interp(eta, self.t, self.plain_tab)

    def p1(self, eta):
        return self.c + (1 - self.c) * self.plain(eta)

    def loglik(self, eta, k, n):
        """log P(k successes, n-k failures on distinct new items | eta)."""
        p = np.clip(self.p1(eta), 1e-300, 1)
        q = np.clip(1 - self.p1(eta), 1e-300, 1)
        return k * np.log(p) + (n - k) * np.log(q)


def gauss_grid(mean, sd, n=2001, width=10):
    x = np.linspace(mean - width * sd, mean + width * sd, n)
    lw = -0.5 * ((x - mean) / sd) ** 2
    return x, lw


def t_grid(mean, scale, nu, n=20001, span=80):
    x = np.linspace(mean - span, mean + span, n)
    lw = -0.5 * (nu + 1) * np.log1p(((x - mean) / scale) ** 2 / nu)
    return x, lw


def logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else float(out.ravel()[0])


def model_pred(h, prior=None, **flags):
    return HierPredictor(prior, h, **flags)
