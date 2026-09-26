"""The hierarchical predictor (paiec/hier.py) and its offline prior (paiec/prior.py)."""
import io
import json
import math
import sys
import time

import numpy as np
import pytest

from paiec import hier as H
from paiec import official as O
from paiec import predict as PH
from paiec import prior as PR
from paiec.hier import (Hyper, HierPredictor, Problem, _Items, expect_sig, item_groups, make_hier,
                        mcq_text, parse_features, select_keys)
from paiec.mcq import floor_of
from paiec.subjects import Spec, attrs, design_row
from tests.synth import make_pairs

sig = lambda x: 0.5 * (1 + np.tanh(0.5 * np.asarray(x, float)))
#: moderate spreads, so simulated probabilities fill every bin
SIM = Hyper(mu0=-0.5, sigma_mu=1.0, sigma_theta=0.3, sigma_delta=1.0, sigma_attr=0.5,
            sigma_d=1.5, sigma_g=1.0)
#: E over a standard normal on a grid far finer than hier's
ZF = np.linspace(-12, 12, 48_001)
WF = np.exp(-ZF ** 2 / 2) / np.exp(-ZF ** 2 / 2).sum()


def subject(name, **kw):
    s = dict.fromkeys(("normalized_name", "provider", "release_date", "access_date", "harness",
                       "harness_version", "reasoning_effort", "subject_features_extra"), "")
    s.update(normalized_name=name, **kw)
    return s


def item(j, bid, features="", text=None):
    return {"item_content": text or f"question {j} of {bid}", "item_features": features,
            "interactors": "", "benchmark_id": bid}


def mcq(j, bid="benchmark_7"):
    return item(j, bid, text=f"Question {j}?\nA) one\nB) two\nC) three\nD) four")


def toy_prior():
    spec = Spec(["openai", "anthropic"], ["high"], 500.0, 3.0)
    coef = np.array([0.1, 0.8, 0.0, 0.2, 0.1, -0.3, 0.3, 0.2, 0.1, 0.4, -0.2, 0.3])
    table = {"gizmo 2": [0.9, 0.6, 2]}
    return PR.SubjectPrior(coef, spec, 0.01 * np.eye(len(coef)), table, {"raw": 2.0, "resid": 2.2},
                           {"benchmarks": list(H.PUBLIC[:4]), "included": list(H.PUBLIC),
                            "excluded": []})


def random_problem(seed=0, n_obs=40, p=6, n_items=15, nu=None, floor=False):
    rng = np.random.default_rng(seed)
    cols = rng.integers(0, p, (n_obs, 3))
    vals = rng.normal(0, 1, (n_obs, 3))
    items = np.concatenate([np.arange(n_items), rng.integers(0, n_items, n_obs - n_items)])
    y = (rng.random(n_obs) < 0.5).astype(float)
    c = np.where(items % 3 == 0, [0.25, 0.5, 0.2][seed % 3], 0.0) if floor else None
    return Problem(cols, vals, y, items, rng.uniform(1, 8, n_items), rng.normal(0, 1, p),
                   rng.uniform(0.5, 3, p), nu, c, rng.normal(0, 0.5, n_obs)), rng.normal(0, 1, p)


def one_pair_exact(h, c, k, n, others=()):
    """The exact posterior predictive for a new item of one pair with k
    successes in n labels on distinct new items (no features, no attribute
    prior), with floor c, by quadrature over the level mu (Gaussian, or
    Student-t with h.nu_mu > 0) and the pair's deviation d; `others` are
    (k, n) counts of other subjects on the same benchmark."""
    VD = h.sigma_theta ** 2 + h.sigma_attr ** 2 + h.sigma_delta ** 2
    S = h.sigma_d ** 2 + h.sigma_g ** 2
    span = 60 if h.nu_mu > 0 else 10 * h.sigma_mu
    mu = np.linspace(h.mu0 - span, h.mu0 + span, 4001)
    d = np.linspace(-8 * math.sqrt(VD), 8 * math.sqrt(VD), 801)
    if h.nu_mu > 0:
        pm = (1 + (mu - h.mu0) ** 2 / (h.nu_mu * h.sigma_mu ** 2)) ** (-(h.nu_mu + 1) / 2)
    else:
        pm = np.exp(-(mu - h.mu0) ** 2 / (2 * h.sigma_mu ** 2))
    pd = np.exp(-d ** 2 / (2 * VD))
    z = np.linspace(-9, 9, 241)
    w = np.exp(-z ** 2 / 2) / np.exp(-z ** 2 / 2).sum()
    Lt = np.linspace(mu[0] + d[0] - 1, mu[-1] + d[-1] + 1, 12001)
    Hf = np.interp(mu[:, None] + d[None, :], Lt, c + (1 - c) * (sig(Lt[:, None] - math.sqrt(S) * z) @ w))
    for ko, no in others:
        pm = pm * ((Hf ** ko * (1 - Hf) ** (no - ko)) * pd[None, :]).sum(1)
    wgt = pm[:, None] * pd[None, :] * Hf ** k * (1 - Hf) ** (n - k)
    return float(np.sum(wgt * Hf) / np.sum(wgt))


# --- the collapsed posterior ---------------------------------------------------------

@pytest.mark.parametrize("nu,floor", [(None, False), ([3.0, 0, 0, 1.5, 0, 0], False),
                                      (None, True), ([3.0, 0, 0, 0, 0, 0], True)])
def test_gradient_and_hessian_match_finite_differences(nu, floor):
    """Single- and multi-label items, fixed offsets, off the mode, with Gaussian
    and Student-t priors, with and without guessing floors (whose items take
    the fixed grid). What is left is the quadrature's own inconsistency: its
    adaptive nodes move with eta, which costs up to ~1e-5 relative on the
    gradient at 20 nodes. On a t coordinate Newton's curvature is the scale
    mixture's, above the exact one, and the Laplace covariance takes the exact
    one where it is positive."""
    for seed in range(3):
        prob, x = random_problem(seed, nu=nu, floor=floor)
        x = x * (1 if nu is None else 2 if floor else 3)   # far enough out that a t curvature turns
        assert len(prob.multi) and prob.single.any()
        assert (prob.floors is not None) == floor == any(part.grid for part in prob.parts)
        st = prob.state(x)
        g, P, P_exact = prob.grad(st), prob.precision(st), prob.precision(st, exact=True)
        h, eye = 1e-5, np.eye(prob.p)
        g_fd = np.array([(prob.state(x + h * e).lp - prob.state(x - h * e).lp) / (2 * h)
                         for e in eye])
        H_fd = np.array([(prob.grad(prob.state(x + h * e)) - prob.grad(prob.state(x - h * e)))
                         / (2 * h) for e in eye])
        assert np.max(np.abs(g - g_fd)) < 3e-5 * max(1.0, np.max(np.abs(g)))
        assert np.max(np.abs(-P_exact - H_fd)) < 1e-3 * np.max(np.abs(P_exact))
        d = np.diag(P - P_exact)
        assert np.all(d >= -1e-12) and (nu is None) == np.all(d == 0)
        _, _, w, c = prob.prior(x)
        lap = np.diag(prob.precision(st, laplace=True))
        assert np.allclose(lap - np.diag(P_exact), np.where(c > 0, 0.0, w - c), rtol=0, atol=1e-12)
        if not floor:
            assert np.linalg.eigvalsh(P).min() > 0
        st, P = prob.solve()
        assert np.max(np.abs(prob.grad(st))) < 1e-6 and prob.converged


def test_pair_and_dense_score_covariances_agree():
    """The two ways precision() subtracts the within-item score covariance, on
    both quadrature parts."""
    prob, x = random_problem(4, n_obs=80, p=9, n_items=20, floor=True)
    st = prob.state(x)
    cnt = np.bincount(prob.item, minlength=prob.n_items)
    assert sorted(part.grid for part in prob.parts) == [False, True]
    out = []
    for dense in (True, False):
        prob.plans = [prob.plan(part, cnt, dense) for part in prob.parts]
        assert all(pl is not None and pl.dense == dense for pl in prob.plans)
        out.append(prob.precision(st))
    assert np.max(np.abs(out[0] - out[1])) < 1e-12 * np.max(np.abs(out[0])) + 1e-12
    plans = prob.plans
    assert sum(len(pl.po) for pl in plans) == sum(n * (n - 1) for n in np.bincount(prob.item) if n > 1)


def _brute_logL(eta, y, c, s2):
    e = np.linspace(-90, 90, 360_001)
    f = -e ** 2 / (2 * s2) - 0.5 * np.log(2 * np.pi * s2)
    for et, yy, cc in zip(eta, y, c):
        z = et - e
        if yy:
            f = f + (np.logaddexp(np.log(cc), np.log1p(-cc) - np.logaddexp(0, -z)) if cc
                     else -np.logaddexp(0, -z))
        else:
            f = f + np.log1p(-cc) - np.logaddexp(0, z)
    mx = f.max()
    return mx + np.log(np.trapz(np.exp(f - mx), e))


def test_item_likelihood_is_the_integral():
    """log L_i against a fine grid: plain items (20 adaptive Gauss-Hermite
    nodes, ~1e-6) and floored ones, including the step-2 reviews' worst cases,
    where a success may be a guess on a hard item or an answer on an easy one
    and the integrand has two modes (20 nodes at one of them were off by up to
    0.04). A floored success puts its item on the 201-node grid."""
    eta = np.array([1.3, -0.4, 0.2, 2.0, -1.0, -3.0, 0.5, -2.5])
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    item_of = np.array([0, 1, 1, 1, 1, 2, 3, 3])
    c = np.array([0, 0, 0, 0, 0, 0.25, 0.5, 0.5])
    s2e = np.array([6.0, 3.0, 8.0, 5.0])
    it = _Items(eta, y, item_of, 4, s2e, floor=c)
    for i in range(4):
        o = item_of == i
        assert it.logL[i] == pytest.approx(_brute_logL(eta[o], y[o], c[o], s2e[i]), abs=1e-5)
    cases = [((-5.16, -3.93), (0, 1), 9.51), ((-3, -3), (1, 0), 9.0), ((-4, -2), (1, 0), 25.0),
             ((0, 0), (1, 0), 50.0), ((-6, -6), (1, 0), 30.0), ((2, -5), (1, 0), 20.0),
             ((-8, -8), (1, 0), 60.0), ((-1.3,) * 31, (1,) * 8 + (0,) * 23, 9.51)]
    rng = np.random.default_rng(0)
    for _ in range(60):
        k = int(rng.integers(1, 4))
        cases.append((tuple(rng.normal(-1.3, 3.0, k)), tuple((rng.random(k) < 0.4).astype(int)), 9.51))
    worst = {True: 0.0, False: 0.0}
    for et, yy, s2 in cases:
        et, yy = np.array(et, float), np.array(yy, float)
        cc = np.full(len(et), 0.25)
        it = _Items(et, yy, np.zeros(len(et), np.int64), 1, np.array([s2]), floor=cc)
        grid = it.parts[0].grid
        assert grid == (yy.max() > 0)
        worst[grid] = max(worst[grid], abs(it.logL[0] - _brute_logL(et, yy, cc, s2)))
    # the grid is good to 1e-5; adaptive nodes on failures far out on one side
    # of the prior (a soft step times the prior) to ~1e-4 here
    assert worst[True] < 1e-5 and 1e-5 < worst[False] < 5e-4


def test_gauss_hermite_error_does_not_reach_predictions(monkeypatch):
    """20 adaptive nodes put log L up to 1e-2 off on one-sided items under a
    wide prior (s2e 25); on a simulated run at item sd 5, about twice the
    public, 48 nodes change no prediction by more than 5e-4 (3e-4 measured;
    on real formative and dense checkpoints 5e-5, step-2 fix notes)."""
    rng = np.random.default_rng(7)
    wide = Hyper(**{**SIM.to_dict(), "sigma_d": 5.0})
    labeled, targets = simulate(rng, wide)
    before = [HierPredictor(None, wide).predict(t, labeled) for t, _ in targets[:150]]
    T, W = np.polynomial.hermite.hermgauss(48)
    for name, value in (("GH", 48), ("_T", T), ("_W", W), ("_LW", np.log(W) + T ** 2)):
        monkeypatch.setattr(H, name, value)
    after = [HierPredictor(None, wide).predict(t, labeled) for t, _ in targets[:150]]
    assert max(abs(a - b) for a, b in zip(before, after)) < 5e-4


def test_floored_fits_converge():
    """Twelve pairs of 31 four-option labels at rates near guessing, subjects
    k, k+5 and k+10 sharing items, with the hard floor: with 20 adaptive nodes
    the line search gave up in 24 of 40 such fits (step-2 review); on the
    grid every fit reaches a Newton decrement below 1e-8."""
    subj = lambda k: subject(f"m{k}", provider="openai")
    h = Hyper(guess=1.0)
    for seed in range(8):
        rng = np.random.default_rng(seed)
        rate = [0.3, 0.5, 0.15, 0.7][seed % 4]
        lab = [[[subj(k), mcq(1000 * (k % 5) + j, f"b{k % 5}")], int(rng.random() < rate)]
               for k in range(12) for j in range(31)]
        model = HierPredictor(None, h)
        fit = model.fit_for(lab)
        assert fit.converged and fit.prob.decrement < 1e-8, seed
        assert model.unconverged == 0


def test_item_residual_is_integrated_not_maximised():
    """160 successes in 200 one-label items: a joint mode over the item effects
    would say 0.70 for a new item; integrating them gives the 0.8 it is."""
    s = subject("solo")
    labeled = [[[s, item(j, "benchmark_0")], int(j < 160)] for j in range(200)]
    h = Hyper(sigma_delta=math.sqrt(4.4), sigma_mu=0.0001, mu0=0.0, sigma_theta=0.0001,
              sigma_attr=0.0001, sigma_d=math.sqrt(8.7), sigma_g=0.0)
    p = HierPredictor(None, h, floor=False, slip=False).predict([s, item(999, "benchmark_0")], labeled)
    assert p == pytest.approx(0.8, abs=0.01)
    # the joint mode over (standing, 200 item effects), with its Laplace variance
    y, s2, vt = (np.arange(200) < 160).astype(float), 8.7, 4.4 + 3e-8
    x = np.zeros(201)
    for _ in range(50):
        q = sig(x[0] - x[1:])
        w = q * (1 - q)
        g = np.concatenate([[np.sum(y - q) - x[0] / vt], -(y - q) - x[1:] / s2])
        Hm = np.diag(np.concatenate([[w.sum() + 1 / vt], w + 1 / s2]))
        Hm[0, 1:] = Hm[1:, 0] = -w
        x = x + np.linalg.solve(Hm, g)
    v = np.linalg.inv(Hm)[0, 0]
    assert expect_sig(x[0], v + s2) == pytest.approx(0.70, abs=0.01)


def test_item_curvature_is_minus_the_derivative_of_the_score():
    """_Items.curvature(delta): minus d/dt of sum_o delta_o g_o as every eta
    moves by delta t, per item, single- and multi-label, floored or not (the
    item's block of Problem.precision along delta; the line uses it for the
    labels it leaves to second order)."""
    rng = np.random.default_rng(4)
    n, n_items = 60, 18
    item_of = np.concatenate([np.arange(n_items), rng.integers(0, n_items, n - n_items)])
    eta, delta = rng.normal(-0.5, 1.5, n), rng.normal(0, 1, n)
    y = (rng.random(n) < 0.5).astype(float)
    c = np.where(item_of % 4 == 0, 0.25, 0.0)
    s2e = rng.uniform(1, 9, n_items)

    def score(t):
        it = _Items(eta + delta * t, y, item_of, n_items, s2e, floor=c)
        return np.bincount(item_of, delta * it.g, n_items)
    it = _Items(eta, y, item_of, n_items, s2e, floor=c)
    assert any(part.grid for part in it.parts) and np.bincount(item_of).max() >= 3
    h = 1e-5
    fd = -(score(h) - score(-h)) / (2 * h)
    assert np.max(np.abs(it.curvature(delta) - fd)) < 1e-3 * np.max(np.abs(fd))


def test_expectation_of_the_sigmoid_is_exact():
    """expect_sig against a far finer grid, over the means and variances the
    model reaches, where the probit approximation it replaced is off by more
    than 0.01."""
    worst, probit = 0.0, 0.0
    for v in (0.0, 0.3, 4.0, 9.0, 20.0, 40.0):
        for m in np.linspace(-12, 12, 49):
            exact = float(sig(m + math.sqrt(v) * ZF) @ WF)
            worst = max(worst, abs(expect_sig(m, v) - exact))
            probit = max(probit, abs(float(sig(m / math.sqrt(1 + math.pi * v / 8))) - exact))
    assert worst < 1e-8 and probit > 0.01


def test_a_target_on_a_labeled_item_reads_its_residual_node_by_node():
    """Five subjects failed one item; a sixth, at the same pinned level, is
    predicted on it. Its residual's posterior is skewed: E[sig(eta - e)] under
    it is 0.0372, and a Gaussian with the same mean and variance says 0.0472
    (the step-2 review measured up to 0.019). Read node by node it is exact."""
    h = Hyper(mu0=-3.0, sigma_mu=1e-4, sigma_theta=1e-4, sigma_attr=1e-4, sigma_delta=1e-4,
              sigma_d=math.sqrt(9.51), sigma_g=0.0)
    labeled = [[[subject(f"m{k}"), item(0, "b")], 0] for k in range(5)]
    e = np.linspace(-80, 80, 160_001)
    f = -e ** 2 / (2 * 9.51) + 5 * -np.logaddexp(0, -(e + 3.0))
    q = np.exp(f - f.max())
    exact = float(q @ sig(-3.0 - e) / q.sum())
    model = HierPredictor(None, h, floor=False, slip=False)
    assert model.predict([subject("new"), item(0, "b")], labeled) == pytest.approx(exact, abs=2e-4)
    it = model.fit_for(labeled).post.it
    assert abs(expect_sig(-3.0 - it.ebar[0], it.evar[0]) - exact) > 0.009


# --- simulated from the model ---------------------------------------------------------

def simulate(rng, h, n_b=3, n_s=5, n_i=40, levels=4):
    """One run drawn from the model's own prior (no attribute prior): labeled,
    and the held-out cells as (input, outcome)."""
    subjects = [subject(f"model {s}") for s in range(n_s)]
    theta = rng.normal(0, math.hypot(h.sigma_theta, h.sigma_attr), n_s)
    labeled, targets = [], []
    for b in range(n_b):
        bid = f"benchmark_{b}"
        mu = rng.normal(h.mu0, h.sigma_mu)
        u = rng.normal(0, h.sigma_g, levels)
        grp = rng.integers(0, levels, n_i)
        e = rng.normal(0, h.sigma_d, n_i)
        items = [item(j, bid, f"tier=t{grp[j]}") for j in range(n_i)]
        for s in range(n_s):
            y = rng.random(n_i) < sig(mu + theta[s] + rng.normal(0, h.sigma_delta) - u[grp] - e)
            perm = rng.permutation(n_i)
            k = int(rng.integers(0, 16))
            labeled += [[[subjects[s], items[j]], int(y[j])] for j in perm[:k]]
            targets += [([subjects[s], items[j]], int(y[j])) for j in perm[k:]]
    return labeled, targets


def test_calibrated_on_data_simulated_from_the_model():
    rng = np.random.default_rng(0)
    pred, out = [], []
    for _ in range(25):
        labeled, targets = simulate(rng, SIM)
        model = HierPredictor(None, SIM, floor=False, slip=False)
        fit = model.fit_for(labeled)
        for (s, it), y in targets:
            w, m, v, s2t = fit.mixture(s, it)
            pred.append(model.combine((w, m, v), s2t, 0.0, 0.0))
            out.append(y)
    pred, out = np.array(pred), np.array(out)
    assert abs(pred.mean() - out.mean()) < 0.015
    bins = np.minimum((pred * 10).astype(int), 9)
    for b in range(10):
        sel = bins == b
        if sel.sum() >= 300:
            assert abs(pred[sel].mean() - out[sel].mean()) < 0.04, b
    brier = np.mean((pred - out) ** 2)
    assert brier < np.mean((out.mean() - out) ** 2)


def test_laplace_error_at_low_budgets():
    """One pair, k of n labels on new items, against the exact posterior
    predictive. The Laplace (Gaussian) posterior of x is 0.013 to 0.02 too
    close to the prior after 1 to 7 labels at the default widths, because the
    mode of a skewed logistic posterior sits nearer the prior than its mean;
    along the target's line (Flags.line) it is exact for one pair."""
    h, s = Hyper(), subject("solo")
    gauss = []
    for n, k in ((1, 0), (1, 1), (3, 0), (3, 3), (7, 7), (31, 31)):
        exact = one_pair_exact(h, 0.0, k, n)
        labeled = [[[s, item(j, "b")], int(j < k)] for j in range(n)]
        ps = [HierPredictor(None, h, floor=False, slip=False, line=line).predict([s, item(999, "b")], labeled)
              for line in (True, False)]
        assert ps[0] == pytest.approx(exact, abs=5e-4), (n, k)
        prior_side = np.sign(HierPredictor(None, h, slip=False).predict([s, item(999, "b")], []) - exact)
        assert np.sign(ps[1] - exact) == prior_side, (n, k)
        gauss.append(abs(ps[1] - exact))
    assert 0.012 < max(gauss[:2]) < 0.017 and 0.017 < max(gauss[2:5]) < 0.023


def test_the_guessing_floor_is_counted_once():
    """A four-option benchmark whose truth is c + (1 - c) sig(eta): with the
    floor in the likelihood, one pair's prediction for a new item is the
    exact posterior predictive, at the hard floor (guess 1, c = 0.25) and the
    default (guess 0.5). Fitting a plain logistic and adding the floor again
    at prediction was 0.06 to 0.15 too high below 90% successes."""
    s = subject("solo")
    for h in (Hyper(guess=1.0), Hyper()):
        c = 0.25 * h.guess
        for n, k in ((7, 6), (15, 10), (31, 10), (31, 20), (31, 28)):
            exact = one_pair_exact(h, c, k, n)
            labeled = [[[s, mcq(j)], int(j < k)] for j in range(n)]
            once = HierPredictor(None, h, slip=False).predict([s, mcq(999)], labeled)
            assert once == pytest.approx(exact, abs=2e-3), (h.guess, n, k)
            if h.guess == 1 and k < 0.9 * n:
                twice = c + (1 - c) * HierPredictor(None, h, slip=False, floor=False).predict(
                    [s, mcq(999)], labeled)
                assert twice - exact > 0.05, (n, k)
    model = HierPredictor(None, Hyper())
    assert model.fit_for([[[s, mcq(0)], 1]]).prob.floors is not None
    assert HierPredictor(None, Hyper(), floor=False).fit_for([[[s, mcq(0)], 1]]).prob.floors is None


def test_the_soft_floor_lets_failures_go_below_one_over_n():
    """31 failures on four-option items: the hard floor holds the next one at
    0.25 at least; with guess = 0.5 it goes down toward 0.125."""
    s = subject("solo")
    labeled = [[[s, mcq(j)], 0] for j in range(31)]
    hard = HierPredictor(None, Hyper(guess=1.0)).predict([s, mcq(99)], labeled)
    soft = HierPredictor(None, Hyper()).predict([s, mcq(99)], labeled)
    assert hard >= 0.25 and 0.125 <= soft < 0.2


def test_mcq_text_reads_the_same_floor_in_linear_time():
    """mcq_text keeps every option mcq.floor_of found (the public items are
    checked in the step-2 fix notes) and makes the quadratic cases linear."""
    texts = ["Question?\nA) one\nB) two\nC) three\nD) four", "Q\n\n   (A) x\n\n  (B) y\n (C) z\n",
             "Pick: (A) 1 (B) 2 (C) 3 (D) 4 (E) 5", "A: one\nB: two\nC: three", "no options here",
             "x\r\nA. a\r\nB. b\r\nC. c\r\n", "end (A) (B) (C)", "A)\nB)\nC)", "", None,
             "  \n \t\n A) a\n\n\n B) b \n C) c", "A)1\nB)2\nC)3"]
    for t in texts:
        assert floor_of(mcq_text(t)) == floor_of(t or ""), t
    for bad in ("\n" * 200_000, " \n" * 100_000, "(" * 200_000):
        t0 = time.perf_counter()
        floor_of(mcq_text(bad))
        assert time.perf_counter() - t0 < 0.5


# --- properties of predict -------------------------------------------------------------

def test_b0_is_the_prior_prediction():
    prior, h = toy_prior(), Hyper()
    model = HierPredictor(prior, h)
    s = subject("Gizmo-2 (high)", provider="OpenAI", release_date="2025-06-01")
    it = item(0, "benchmark_9", "tier=a", "Pick one\nA) x\nB) y\nC) z\nD) w")
    mt, vt = model.theta_prior(b"any", s)
    v = h.sigma_mu ** 2 + vt + h.sigma_delta ** 2 + h.sigma_d ** 2 + h.sigma_g ** 2
    c = 0.25 * h.guess
    want = c + (1 - c - h.slip) * float(sig(h.mu0 + mt + math.sqrt(v) * ZF) @ WF)
    assert model.predict([s, it], []) == pytest.approx(want, abs=1e-9)
    assert model.predict([s, it], None) == model._fallback([s, it])
    m_a, u = prior.attribute(s)
    assert mt != m_a            # the identity table moved it, within its cap
    assert abs(mt - m_a) <= h.id_cap * abs(prior.table["gizmo 2"][1]) + 1e-12


def test_more_successes_raise_the_prediction():
    others = [[[subject(f"other {k}"), item(j, "benchmark_1")], int((j + k) % 3 == 0)]
              for k in range(4) for j in range(10)]
    target = [subject("target"), item(99, "benchmark_1")]
    ps = []
    for k in range(9):
        own = [[[subject("target"), item(20 + j, "benchmark_1")], int(j < k)] for j in range(8)]
        ps.append(make_hier()(target, others + own))
    assert all(a < b for a, b in zip(ps, ps[1:]))


def test_pure_function_of_input_and_labeled():
    rng = np.random.default_rng(3)
    labeled, targets = simulate(rng, SIM)
    shuffled = [labeled[i] for i in rng.permutation(len(labeled))]
    a, b = HierPredictor(toy_prior()), HierPredictor(toy_prior())
    first = [a.predict(t, labeled) for t, _ in targets[:40]]
    a.predict(targets[0][0], labeled[:5])          # another fit in the cache between
    assert [a.predict(t, shuffled) for t, _ in targets[:40]] == first
    assert [b.predict(t, labeled) for t, _ in targets[:40]] == first
    assert [b.predict(t, labeled) for t, _ in reversed(targets[:40])] == first[::-1]
    assert a.failures == b.failures == 0


def test_subject_fields_are_read_as_the_text_their_key_hashes():
    """release_date 2024 and '2024' share a subject key, so they must share a
    prior, whichever of them a worker meets first."""
    s_str, s_int = subject("gizmo", release_date="2024"), subject("gizmo", release_date=2024)
    assert PH.subject_key(s_str) == PH.subject_key(s_int)
    labeled = [[[subject(f"m{k}"), item(j, "benchmark_1")], (j + k) % 2] for k in range(3)
               for j in range(5)]
    target = item(99, "benchmark_1")
    a, b = HierPredictor(toy_prior()), HierPredictor(toy_prior())
    pa = [a.predict([s_str, target], labeled), a.predict([s_int, target], labeled)]
    pb = [b.predict([s_int, target], labeled), b.predict([s_str, target], labeled)]
    assert pa[0] == pa[1] == pb[0] == pb[1]


def test_identical_under_the_platforms_copies_and_workers():
    pairs = make_pairs(3, 2, 100)
    fast = O.run_official(pairs, lambda: make_hier(), deepcopy=False, workers=1)
    slow = O.run_official(pairs, lambda: make_hier(), deepcopy=True, workers=5)
    assert [r["brier"] for r in fast["rows"]] == [r["brier"] for r in slow["rows"]]


def test_unseen_components_stay_at_their_prior_variance():
    """A component no label touched adds its prior variance to v and nothing
    else, and a wider prior pulls the prediction toward 0.5. (The Gaussian
    path, where v is exactly the sum; the line only reshapes what labels
    touched.)"""
    labeled = [[[subject(f"m{k}"), item(j, "benchmark_1", f"tier=t{j % 3}")], int(j % 2)]
               for k in range(3) for j in range(12)]
    h = Hyper(mu0=-2.0)
    model = HierPredictor(None, h, line=False)
    fit = model.fit_for(labeled)
    s = subject("m0")
    cols = [fit.col[("mu", "benchmark_1")], fit.col[("th", ("n", "m0"))],
            fit.col[("de", "benchmark_1", PH.subject_key(s))]]
    a_cov = fit.post.cov[np.ix_(cols, cols)].sum()
    m, v, _, _, _ = model.components([s, item(50, "benchmark_1", "tier=t9")], labeled)
    assert m == pytest.approx(fit.post.x[cols].sum())
    assert v == pytest.approx(a_cov + h.sigma_g ** 2 + h.sigma_d ** 2)   # one key: all of sigma_g
    m2, v2, _, _, _ = model.components([s, item(51, "benchmark_1", "tier=t8")], labeled)
    assert (m2, v2) == (m, v)
    _, v_seen, _, _, _ = model.components([s, item(52, "benchmark_1", "tier=t1")], labeled)
    assert v - v_seen > 0.8 * h.sigma_g ** 2
    lined = HierPredictor(None, h)
    _, v_line, _, _, _ = lined.components([s, item(50, "benchmark_1", "tier=t9")], labeled)
    assert v_line == pytest.approx(v, rel=0.2)
    target = [s, item(0, "benchmark_2")]                 # a new benchmark
    for field in ("sigma_mu", "sigma_delta", "sigma_d", "sigma_g"):
        ps = []
        for scale in (0.5, 1.0, 2.0, 4.0):
            wide = Hyper(**{**h.to_dict(), field: getattr(h, field) * scale})
            ps.append(HierPredictor(None, wide).predict(target, []))
        assert all(abs(a - 0.5) > abs(b - 0.5) for a, b in zip(ps, ps[1:])), field
    ps = [HierPredictor(None, Hyper(**{**h.to_dict(), "sigma_mu": sd})).predict(target, labeled)
          for sd in (0.5, 1.0, 2.0, 4.0)]
    assert all(abs(a - 0.5) > abs(b - 0.5) for a, b in zip(ps, ps[1:]))


def test_a_level_one_labeled_item_carries_is_folded_into_it():
    """A key with a value per labeled item is the same generative model as no
    key (a sum of two Gaussians per item), so it must give the same answer;
    as a column in the joint mode it pulled every answer ~0.01 toward 0.5. A
    target sharing a folded level reads it back from its carrier."""
    s, h = subject("solo"), Hyper()
    for k1, k0 in ((1, 0), (7, 0), (15, 0), (3, 12), (24, 7)):
        ps = []
        for feat in (lambda j: "", lambda j: f"uid=q{j}"):
            labeled = [[[s, item(j, "b", feat(j))], 1] for j in range(k1)] + \
                      [[[s, item(100 + j, "b", feat(100 + j))], 0] for j in range(k0)]
            model = HierPredictor(None, h, floor=False, slip=False)
            ps.append(model.predict([s, item(999, "b", feat(999))], labeled))
        assert ps[1] == pytest.approx(ps[0], abs=1e-9), (k1, k0)
        assert not any(k[0] == "g" for k in model.fit_for(labeled).col)
    # three subjects answer one hard item of tier a; tier b has two items
    others = [[[subject(f"o{k}"), item(0, "b", "tier=a")], 0] for k in range(3)] + \
             [[[subject(f"o{k}"), item(j, "b", "tier=b")], 1] for k in range(3) for j in (1, 2)]
    model = HierPredictor(None, h)
    fit = model.fit_for(others)
    assert ("g", "b", "tier", "a") in fit.folded and ("g", "b", "tier", "b") in fit.col
    target = subject("new")
    pa = model.predict([target, item(7, "b", "tier=a")], others)
    pz = model.predict([target, item(8, "b", "tier=z")], others)
    assert pa < pz
    m_a, v_a, *_ = model.components([target, item(7, "b", "tier=a")], others)
    m_z, v_z, *_ = model.components([target, item(8, "b", "tier=z")], others)
    assert v_a < v_z


def test_a_key_keeps_at_most_max_levels_columns():
    """Past MAX_LEVELS levels of one key the rarest share nothing: each
    carrier's residual takes the level's variance, a target on another item
    of such a level gets it as unseen."""
    bid = "b"
    items = [item(j, bid, f"topic=v{j % 100}") for j in range(300)]
    labeled = [[[subject(f"m{k}"), items[j]], int((j + k) % 3 == 0)] for k in range(3)
               for j in range(300)]
    model = HierPredictor(None, Hyper())
    fit = model.fit_for(labeled)
    assert sum(k[0] == "g" for k in fit.col) == H.MAX_LEVELS
    assert len(fit.dropped) == 100 - H.MAX_LEVELS
    lev = sorted(fit.dropped)[0]
    j = int(lev[3][1:])
    p = model.predict([subject("new"), item(999, bid, f"topic=v{j}")], labeled)
    assert 0 < p < 1 and model.failures == 0


def test_delta_off_keeps_each_dicts_attribute_offset():
    """Two dicts of one canonical name with different attributes share theta;
    with the pair deviation off, each keeps its own attribute offset."""
    lo, hi = subject("gizmo", provider="openai"), subject("gizmo", provider="openai",
                                                         reasoning_effort="high")
    labeled = [[[lo, item(j, "benchmark_1")], j % 2] for j in range(10)] + \
              [[[hi, item(20 + j, "benchmark_1")], j % 2] for j in range(10)]
    model = HierPredictor(toy_prior(), delta=False)
    m_lo, *_ = model.components([lo, item(99, "benchmark_1")], labeled)
    m_hi, *_ = model.components([hi, item(99, "benchmark_1")], labeled)
    offset = model.theta_prior(b"h", hi)[0] - model.theta_prior(b"l", lo)[0]
    assert offset > 0.2 and m_hi - m_lo == pytest.approx(offset, abs=1e-9)


@pytest.mark.parametrize("flags", [
    {}, {"identity": False}, {"attributes": False}, {"pool_mu": False}, {"link": False},
    {"delta": False}, {"groups": False}, {"prefix_groups": True}, {"text": True},
    {"floor": False, "slip": False}, {"subject_key": "full"}, {"subject_key": "name"},
    {"line": False}, {"hyper": Hyper(sigma_mu=1.72, nu_mu=3.0)},
    {"hyper": Hyper(sigma_mu=1.72, nu_mu=3.0), "t_mixture": False},
])
def test_every_ablation_runs_the_official_protocol(flags):
    """Three benchmarks, two with features; at budget 31 each holds 93 labeled
    items, past the text term's warm-up. (tests.synth draws an item's difficulty
    afresh for every subject, so pooling items across subjects cannot pay here:
    the check is that every variant runs, not what it scores.)"""
    pairs = make_pairs(3, 3, 100)
    for p in pairs:
        if p.benchmark_id == "b2":
            for r in p.responses:
                r.item["item_features"] = ""
    made, flags = [], dict(flags)
    hyper = flags.pop("hyper", Hyper())

    def factory():
        made.append(HierPredictor(toy_prior(), hyper, **flags))
        return made[-1].predict
    res = O.run_official(pairs, factory, deepcopy=False)
    assert sum(m.failures for m in made) == 0
    assert sum(m.unconverged for m in made) == 0
    assert 0 < res["brier"]["ALC"] < 0.3
    if flags.get("text"):
        assert any(m._fits and any(f.text for f in m._fits.values()) for m in made)
    if hyper.nu_mu > 0 and flags.get("t_mixture", True):
        assert any(f.mix for m in made for f in m._fits.values())


def test_zero_labels_all_zero_all_one_and_a_single_subject():
    s, bid = subject("solo"), "benchmark_5"
    target = [s, item(99, bid)]
    model = HierPredictor(None, Hyper())
    p0 = model.predict(target, [])
    lo = model.predict(target, [[[s, item(j, bid)], 0] for j in range(31)])
    hi = model.predict(target, [[[s, item(j, bid)], 1] for j in range(31)])
    assert 1e-4 <= lo < p0 < hi <= 1 - 1e-4
    assert hi > 0.9 and lo < 0.1
    assert model.failures == 0


def test_student_t_fits_iterate_to_convergence():
    """Near its mode a Student-t fit converges linearly (Newton's curvature on a
    t coordinate is the scale mixture's). Stopped after six close steps, one
    subject 31/31 on one benchmark and 0/1 on another kept a Newton decrement
    of 1.9e-7 and counted as unconverged (final review); iterated on, it
    converges, and the prediction moves by 6e-6."""
    s, h = subject("linked"), Hyper(sigma_mu=1.829, nu_mu=3.0)
    lab = [[[s, item(j, "benchmark_1")], 1] for j in range(31)] + [[[s, item(0, "benchmark_2")], 0]]
    target = [s, item(999, "benchmark_2")]
    model = HierPredictor(None, h, floor=False, slip=False)
    p = model.predict(target, lab)
    assert model.unconverged == 0 and model.fit_for(lab).prob.decrement < 1e-8
    old = H.CLOSE_MAX
    try:
        H.CLOSE_MAX = 6                 # the old rule
        stopped = HierPredictor(None, h, floor=False, slip=False)
        q = stopped.predict(target, lab)
    finally:
        H.CLOSE_MAX = old
    assert stopped.unconverged == 1 and abs(p - q) < 2e-5


def test_student_t_level():
    """The quadrature over an untouched t level is exact for sigmoids, and a
    large nu gives the Gaussian back. A touched t level is its scale mixture:
    against the exact posterior predictive of one pair it is within 0.002
    (the Laplace fit on the t density itself was up to 0.065 off, 0.028 after
    one label), and a new subject after six others' 31 failures each within
    0.01."""
    from paiec.hier import lam_nodes, t_nodes
    T, w = t_nodes(3.0)
    grid = np.linspace(-1, 1, 2_000_001)[1:-1] * np.pi / 2
    dens = np.cos(grid) ** 2 / (np.pi / 2)                  # t3 in T = sqrt(3) tan(u)
    for a, b in [(0.3, 1.0), (-2.0, 2.5), (4.0, 0.5)]:
        exact = np.trapz(sig(a + b * math.sqrt(3) * np.tan(grid)) * dens, grid)
        assert float(w @ sig(a + b * T)) == pytest.approx(exact, abs=1e-8)
    lam, lw = lam_nodes(3.0)
    assert len(lam) == H.LAM and np.exp(lw).sum() == pytest.approx(1.0)
    assert float(np.exp(lw) @ lam) == pytest.approx(1.0, abs=2e-3)      # E[lambda] = 1
    s, bid = subject("new"), "benchmark_1"
    target = [s, item(999, bid)]
    t3h = Hyper(sigma_mu=1.829, nu_mu=3.0)
    t3 = HierPredictor(None, t3h, floor=False, slip=False)
    assert t3.predict(target, []) == t3._fallback(target)
    big = HierPredictor(None, Hyper(nu_mu=300)).predict(target, [])
    assert big == pytest.approx(HierPredictor(None, Hyper()).predict(target, []), abs=0.002)
    solo = subject("solo")
    for n, k in ((1, 0), (1, 1), (3, 3), (7, 7), (31, 0)):
        labeled = [[[solo, item(j, bid)], int(j < k)] for j in range(n)]
        exact = one_pair_exact(t3h, 0.0, k, n)
        assert t3.predict([solo, item(999, bid)], labeled) == pytest.approx(exact, abs=2e-3), (n, k)
        if (n, k) == (3, 3):
            old = HierPredictor(None, t3h, floor=False, slip=False, t_mixture=False, line=False)
            assert old.predict([solo, item(999, bid)], labeled) - exact < -0.05
    many = [[[subject(f"o{m}"), item(100 * m + j, bid)], 0] for m in range(6) for j in range(31)]
    exact = one_pair_exact(t3h, 0.0, 0, 0, others=[(0, 31)] * 6)
    assert t3.predict(target, many) == pytest.approx(exact, abs=0.01)
    assert t3.failures == 0 and t3.unconverged == 0
    with pytest.raises(ValueError):
        HierPredictor(None, t3h, pool_mu=False)


def test_linking_across_benchmarks():
    """Off, a subject's labels on another benchmark change nothing (the
    posterior factorises over benchmarks); on, they move its prediction in
    their direction, the more the larger the link weight. With no attribute
    prior even weight 0 links a little: the spread attributes would have
    explained (sigma_attr) is the subject's on every benchmark alike."""
    others = [[[subject(f"o{k}"), item(j, bid)], int((j + k) % 2)]
              for bid in ("benchmark_1", "benchmark_2") for k in range(3) for j in range(8)]
    s = subject("linked")
    own = [[[s, item(100 + j, "benchmark_1")], 1] for j in range(30)]
    target = [s, item(999, "benchmark_2")]
    h = Hyper()
    assert h.relink(0.4).link_weight == pytest.approx(0.4)
    assert h.relink(0.4).sigma_theta ** 2 + h.relink(0.4).sigma_delta ** 2 == \
        pytest.approx(h.sigma_theta ** 2 + h.sigma_delta ** 2)
    off = HierPredictor(None, h, link=False)
    assert off.predict(target, others + own) == pytest.approx(off.predict(target, others), abs=1e-9)
    ps = [HierPredictor(None, h.relink(w)).predict(target, others + own)
          for w in (0.0, 0.02, 0.3, 0.8)]
    base = HierPredictor(None, h.relink(0.3)).predict(target, others)
    assert off.predict(target, others) < ps[0]
    assert all(a < b for a, b in zip(ps, ps[1:])) and ps[2] > base


def test_a_failed_fit_is_cached_and_falls_back(monkeypatch, capsys):
    """One traceback, one attempt, then the prior-only answer for every target."""
    calls = []

    def boom(self, *a, **k):
        calls.append(1)
        raise np.linalg.LinAlgError("synthetic")
    monkeypatch.setattr(Problem, "solve", boom)
    labeled = [[[subject("a"), item(j, "b")], j % 2] for j in range(6)]
    model = HierPredictor(toy_prior())
    targets = [[subject("a"), item(50 + j, "b")] for j in range(20)]
    ps = [model.predict(t, labeled) for t in targets]
    assert len(calls) == 1 and model.failures == 20
    assert ps == [model.predict(t, []) for t in targets]
    assert capsys.readouterr().err.count("Traceback") == 1


def test_a_closed_stderr_does_not_break_the_fallback(monkeypatch):
    """The first failure's traceback goes to stderr; a closed one must not
    turn the fallback into an exception (the step-2 review saw exit 40)."""
    monkeypatch.setattr(Problem, "solve", lambda self, *a, **k: 1 / 0)
    closed = io.StringIO()
    closed.close()
    monkeypatch.setattr(sys, "stderr", closed)
    model = HierPredictor(None)
    labeled = [[[subject("a"), item(j, "b")], j % 2] for j in range(6)]
    p = model.predict([subject("a"), item(9, "b")], labeled)
    assert p == model.predict([subject("a"), item(9, "b")], []) and model.failures == 1


def test_a_broken_subject_prior_degrades_to_no_attributes():
    """Loading refuses a malformed prior; one built around the check costs the
    attribute prior, not every prediction."""
    good = toy_prior().to_dict()
    for bad in ({**good, "coef": good["coef"][:-1]}, {**good, "cov": [[1.0]]},
                {**good, "table": {"x": [0.1]}}, {**good, "coef": good["coef"][:-1] + [float("nan")]},
                {**good, "spec": {"providers": []}}):
        with pytest.raises(ValueError):
            PR.SubjectPrior.from_dict(json.loads(json.dumps(bad)))
    broken = toy_prior()
    broken.coef = broken.coef[:-1]
    labeled = [[[subject("a"), item(j, "b")], j % 2] for j in range(6)]
    targets = [[subject(f"m{k}", provider="openai"), item(50 + k, "b")] for k in range(5)]
    model, plain = HierPredictor(broken), HierPredictor(None)
    assert [model.predict(t, labeled) for t in targets] == [plain.predict(t, labeled) for t in targets]
    assert model.failures == 0


def test_exclusions_must_match():
    """A leave-benchmarks-out prior needs hyperparameters fitted without the
    same benchmarks: the defaults (fitted on all five) or another exclusion
    set are refused."""
    pairs = make_pairs(4, 3, 60)
    prior = PR.build_prior(pairs, exclude=["b2"])
    with pytest.raises(ValueError):
        HierPredictor(prior)
    with pytest.raises(ValueError):
        HierPredictor(prior, Hyper())
    hyper, _ = PR.fit_hyper(pairs, exclude=["b2"], prior=prior)
    assert hyper.excluded == ("b2",)
    with pytest.raises(ValueError):
        HierPredictor(PR.build_prior(pairs), hyper)
    HierPredictor(prior, hyper)
    HierPredictor(None, hyper)
    assert PR.build(pairs, exclude=["b2"])[1] == hyper
    # pairs filtered beforehand record no exclusion, but what they held
    pre = [p for p in pairs if p.benchmark_id != "b2"]
    prior_pre = PR.build_prior(pre)
    assert prior_pre.meta["excluded"] == [] and prior_pre.meta["included"] == ["b0", "b1"]
    for h in (None, Hyper(), PR.fit_hyper(pairs)[0]):  # the defaults saw PUBLIC; the last saw b2
        with pytest.raises(ValueError):
            HierPredictor(prior_pre, h)
    hyper_pre, _ = PR.fit_hyper(pre, prior=prior_pre)
    assert hyper_pre.included == ("b0", "b1") and hyper_pre.excluded == ()
    HierPredictor(prior_pre, hyper_pre)
    with pytest.raises(ValueError):                      # a prior of other pairs
        PR.fit_hyper(pairs, prior=prior_pre)
    # a prior that does not record what it was fitted on cannot use the defaults
    unrecorded = PR.SubjectPrior.from_dict({**prior_pre.to_dict(), "meta": {"benchmarks": ["b0"],
                                                                          "excluded": []}})
    with pytest.raises(ValueError):
        HierPredictor(unrecorded)
    HierPredictor(toy_prior())                           # fitted on exactly PUBLIC
    assert Hyper().included == H.PUBLIC and Hyper(excluded=["swe_rebench"]).included == H.PUBLIC[:4]
    assert Hyper.from_dict(Hyper().to_dict()) == Hyper()


def test_the_level_prior_needs_three_levels():
    """mu0 and sigma_mu come from three benchmark levels or more. With one or
    two left, their mean is the level of the very benchmarks a strict run-LOBO
    fit left, known with no spread (the final review saw mu0 = -4.14 from
    multi_swebench alone, which cost 0.013 ALC on simulated runs), so both take
    REFERENCE. From n levels a new one spreads by sd sqrt(1 + 1/n), widened for
    the Gaussian level. A single multi-subject benchmark gives sigma_delta from
    its standings' whole variance, flagged as approximate, instead of the
    REFERENCE width."""
    pairs = make_pairs(6, 4, 100)
    for p in pairs:
        p.subject["provider"] = "openai" if p.subject_id < "s3" else "anthropic"
        p.subject["release_date"] = f"2025-0{1 + int(p.subject_id[1:])}-01"
    names = sorted({p.benchmark_id for p in pairs})
    for kept in (names[:1], names[:2]):
        held = [b for b in names if b not in kept]
        prior = PR.build_prior(pairs, held)
        hyper, rep = PR.fit_hyper(pairs, held, prior)
        assert sorted(rep["levels"]) == kept
        assert (hyper.mu0, hyper.sigma_mu) == (0.0, PR.REFERENCE["sigma_mu"])
        assert {"mu0", "sigma_mu"} <= set(rep["fallback"])
        assert all(abs(hyper.mu0 - v) > 1e-3 for v in rep["levels"].values())
        if len(kept) == 1:
            m = prior.meta
            assert m["s2_res"] is None and rep["approximate"] == ["sigma_delta"]
            th2 = PR.REFERENCE["sigma_theta"] ** 2
            assert hyper.sigma_delta == pytest.approx(
                math.sqrt(max(m["total_var"] - m["u_mean"] - th2, 0.1)) * PR.WIDEN[1])
            assert "sigma_delta" in rep["estimated"] and "sigma_attr" in rep["fallback"]
        else:
            assert rep["approximate"] == []
    held = names[3:]
    for nu in (0.0, 3.0):
        hyper, rep = PR.fit_hyper(pairs, held, PR.build_prior(pairs, held), nu_mu=nu)
        lv = np.array(list(rep["levels"].values()))
        assert len(lv) == 3 and hyper.mu0 == pytest.approx(lv.mean())
        assert hyper.sigma_mu == pytest.approx(
            lv.std(ddof=1) * math.sqrt(1 + 1 / 3) * (PR.WIDEN[0] if nu == 0 else 1.0))
    zero, rep = PR.fit_hyper(pairs, names[1:], PR.build_prior(pairs, names[1:]), centre="zero")
    assert (zero.mu0, zero.sigma_mu) == (0.0, PR.REFERENCE["sigma_mu"])


def test_a_line_past_line_max_integrates_the_most_moved_items():
    """A dense checkpoint whose line moves more than LINE_MAX labels by 2% of
    the most moved: the line integrates the most moved items as far as
    LINE_MAX labels go (before the final review it fell back to the Gaussian
    whole, up to 0.05 off on dense B31), and the rest enter to second order,
    their curvature what is left of 1 / sd^2 once the prior's and the
    integrated items' own are taken out: the direct sum over them."""
    rng = np.random.default_rng(8)
    items = [item(j, "benchmark_0") for j in range(120)]
    labeled = [[[subject(f"model {k}"), items[j]], int(rng.random() < 0.35)]
               for k in range(30) for j in rng.choice(120, 31, replace=False)]
    model = HierPredictor(None, Hyper(), floor=False, slip=False)
    fit = model.fit_for(labeled)
    post, prob = fit.post, fit.prob
    tg = fit._target(subject("model 3"), item(999, "benchmark_0"))
    key = tuple(sorted(tg.a.items()))
    cs, av = [c for c, _ in key], np.array([v for _, v in key])
    Sa = post.cov[:, cs] @ av
    s2 = float(Sa[cs] @ av)
    d = Sa / s2
    delta = np.einsum("os,os->o", prob.vals, d[prob.cols])
    moved = np.bincount(prob.item, np.abs(delta) >= H.LINE_FRAC * np.abs(delta).max())
    assert moved[moved > 0].size and np.bincount(prob.item)[moved > 0].sum() > H.LINE_MAX
    line = fit._line(post, tg.a)
    hit = np.array(sorted(line.index))
    labs = fit._labels(hit)
    assert line is not None and 0.8 * H.LINE_MAX < len(labs) <= H.LINE_MAX
    li = np.searchsorted(hit, prob.item[labs])
    it0 = _Items(prob.eta(post.x)[labs], prob.y[labs], li, len(hit), prob.s2e[hit],
                 post.it.mode[hit], prob.c[labs])
    rest = 1 / s2 - it0.curvature(delta[labs]).sum() - \
        H._prior_curvature(post.x, prob.m, post.V, post.nu) @ (d * d)
    full = prob.state(post.x, post.it.mode).it.curvature(delta)
    off = np.ones(len(full), bool)
    off[hit] = False
    assert rest > 0 and rest == pytest.approx(full[off].sum(), rel=1e-6)
    assert 0 < model.predict([subject("model 3"), item(999, "benchmark_0")], labeled) < 1


# --- robustness and speed --------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    None, [], [None, None], "x", [subject("a")], [subject("a"), "item"],
    [{"normalized_name": 3}, {"benchmark_id": None, "item_features": 7}],
    [subject("a"), {"item_content": None, "item_features": "{bad json", "benchmark_id": "b"}],
    [subject("a"), item(0, "b", "a=[1;b=2")],
])
def test_malformed_inputs_never_raise(bad):
    labeled = [[[subject("a"), item(j, "b", "k=v;n=3")], j % 2] for j in range(5)]
    for lab in (labeled, None, [], "junk", [None, 1, [[1, 2], 1]],
                labeled + [[[subject("a"), item(9, "b")], "1"], [[subject("a"), item(8, "b")], 2]]):
        p = make_hier(toy_prior())(bad, lab)
        assert isinstance(p, float) and 1e-4 <= p <= 1 - 1e-4


def test_formative_size_is_fast():
    """12 pairs of 31 labels and 500 targets: fit once, answer each."""
    rng = np.random.default_rng(1)
    labeled, targets = [], []
    for k in range(12):
        bid = f"benchmark_{k % 5}"
        s = subject(f"model {k}", provider="openai")
        for j in range(31):
            labeled.append([[s, item(1000 * k + j, bid, f"topic=t{j % 6};idx={j}")], int(rng.random() < 0.4)])
        targets += [[s, item(1000 * k + 100 + j, bid, f"topic=t{j % 7}")] for j in range(42)]
    predict = make_hier(toy_prior())
    t = time.perf_counter()
    ps = [predict(tg, labeled) for tg in targets[:500]]
    assert time.perf_counter() - t < 5.0
    assert all(1e-4 <= p <= 1 - 1e-4 for p in ps)


@pytest.mark.parametrize("keys", [["lang=l{1}"], ["k{0}=v{1}"] * 8])
def test_dense_size_fits_in_seconds(keys):
    """80 subjects x 31 labels over 2,000 items of one benchmark, 40% of labels
    on items another subject also labeled; with one 8-level key, and with eight
    keys of 60 levels each (hundreds of group columns)."""
    rng = np.random.default_rng(2)
    feats = [";".join(k.format(i, (37 * j + 11 * i * i) % (8 if len(keys) == 1 else 60))
                      for i, k in enumerate(keys)) for j in range(2000)]
    items = [item(j, "benchmark_0", feats[j]) for j in range(2000)]
    shared = rng.choice(2000, 300, replace=False)
    labeled = [[[subject(f"model {k}"), items[j]], int(rng.random() < 0.3)]
               for k in range(80) for j in np.concatenate([rng.choice(shared, 12, replace=False),
                                                          rng.choice(2000, 19, replace=False)])]
    model = HierPredictor(None, Hyper())
    t = time.perf_counter()
    fit = model.fit_for(labeled)
    assert time.perf_counter() - t < 10.0
    assert len(fit.item_idx) > 500 and len(fit.prob.multi) > 250
    assert len(fit.post.x) == 161 + (8 if len(keys) == 1 else 480)
    assert 0 < model.predict([subject("model 3"), items[5]], labeled) < 1


def test_a_long_list_is_fitted_benchmark_by_benchmark():
    """50k labels over five benchmarks (330 pairs each of 31, a 700-level key)
    need more than MAX_COLUMNS columns jointly; split by benchmark, with each
    key capped at MAX_LEVELS columns, they fit in seconds."""
    rng = np.random.default_rng(5)
    labeled = []
    for b in range(5):
        bid = f"benchmark_{b}"
        items = [item(j, bid, f"topic=v{j % 700}") for j in range(3000)]
        for k in range(330):
            s = subject(f"model {b}-{k}")
            labeled += [[[s, items[j]], int(rng.random() < 0.4)] for j in rng.choice(3000, 31, replace=False)]
    assert len(labeled) > 50_000
    model = HierPredictor(None, Hyper())
    t = time.perf_counter()
    fit = model.fit_for(labeled)
    assert time.perf_counter() - t < 30.0
    assert isinstance(fit, H._Split) and len(fit.fits) == 5
    assert all(len(f.post.x) <= H.MAX_COLUMNS for f in fit.fits.values())
    assert all(sum(k[0] == "g" for k in f.col) <= H.MAX_LEVELS for f in fit.fits.values())
    p = model.predict([subject("model 0-3"), item(5, "benchmark_0", "topic=v5")], labeled)
    assert 0 < p < 1 and model.failures == 0


# --- features ---------------------------------------------------------------------------

def test_parse_features_formats():
    assert parse_features("competition=aime_2025;problem_idx=2;prompt_source=user_message") == {
        "competition": "aime_2025", "problem_idx": "2", "prompt_source": "user_message"}
    assert parse_features('image_detail=["high"];competition=kangaroo') == {
        "image_detail": '["high"]', "competition": "kangaroo"}
    assert parse_features('{"lang": "go", "n": 3}') == {"lang": "go", "n": "3"}
    assert parse_features("lang: rust\nsize: big") == {"lang": "rust", "size": "big"}
    assert parse_features("hard;visual") == {"hard": "1", "visual": "1"}
    assert parse_features("just a label") == {"_": "just a label"}
    assert parse_features("") == parse_features("  ") == parse_features(None) == {}
    assert parse_features(7) == {"_": "7"}
    assert list(parse_features("x=" + "y" * 5000)) == ["_"]
    it = {"item_features": "a=1", "interactors": "opponent=x"}
    assert item_groups(it) == {"a": "1", "interactors.opponent": "x"}


def test_key_rule_drops_numeric_and_constant_keys():
    feats = [{"comp": "aime", "idx": "1", "src": "u"}, {"comp": "hmmt", "idx": "2", "src": "u"},
             {"comp": "aime", "idx": "3", "src": "u", "img": '["high"]'}]
    assert select_keys(feats) == ["comp", "img"]         # img: '["high"]' vs '<missing>'
    assert select_keys([{"comp": "aime"}]) == []
    assert select_keys([]) == []
    ids = [{"id": f"x{j}", "tier": "ab"[j % 2]} for j in range(40)]
    assert select_keys(ids) == ["tier"]                  # a level per item names items
    assert select_keys(ids[:20]) == ["tier", "id"]       # too few items to tell


# --- the offline prior ------------------------------------------------------------------

def test_prior_round_trips_through_json_and_respects_exclusions():
    pairs = make_pairs(6, 3, 100)
    for p in pairs:
        p.subject["provider"] = "openai" if p.subject_id < "s3" else "anthropic"
        p.subject["release_date"] = f"2025-0{1 + int(p.subject_id[1:])}-01"
    prior = PR.build_prior(pairs, exclude=["b2"])
    assert prior.meta["benchmarks"] == ["b0", "b1"] and prior.meta["excluded"] == ["b2"]
    assert prior.meta["linked"] == 6                     # every name seen on both benchmarks
    assert all(len(r) == 4 and r[3] <= r[2] for r in prior.table.values())
    back = PR.SubjectPrior.from_dict(json.loads(json.dumps(prior.to_dict())))
    for p in pairs[:6]:
        assert back.attribute(p.subject) == prior.attribute(p.subject)
        assert back.identity(p.subject) == prior.identity(p.subject)
    hyper, rep = PR.fit_hyper(pairs, exclude=["b2"], prior=prior)
    assert set(rep["levels"]) == {"b0", "b1"}
    assert {"sigma_d", "sigma_delta"} <= set(rep["estimated"]) and rep["approximate"] == []
    assert rep["fallback"] == ["mu0", "sigma_mu"]        # two levels make neither
    assert (hyper.mu0, hyper.sigma_mu) == (0.0, PR.REFERENCE["sigma_mu"])
    assert hyper.included == ("b0", "b1") and prior.meta["included"] == ["b0", "b1"]
    assert all(math.isfinite(v) for k, v in hyper.to_dict().items() if k not in ("excluded", "included"))
    assert Hyper.from_dict(json.loads(json.dumps(hyper.to_dict()))) == hyper
    p2, h2 = PR.from_json(json.loads(json.dumps(PR.to_json(prior, hyper))))
    assert h2 == hyper and p2.to_dict() == json.loads(json.dumps(prior.to_dict()))
    assert PR.from_json(PR.to_json(None, hyper))[0] is None
    with pytest.raises(ValueError):
        PR.from_json({"coef": [], "spec": {}})           # the Predictor's prior.json
    with pytest.raises(ValueError):
        PR.from_json({**PR.to_json(None, hyper), "hyper": {**hyper.to_dict(), "sigma_d": -1.0}})
    for k in ("excluded", "included"):
        with pytest.raises(ValueError):
            PR.from_json({**PR.to_json(None, hyper), "hyper": {**hyper.to_dict(), k: "b2"}})
    with pytest.raises(ValueError):                      # a prior built on other exclusions
        PR.fit_hyper(pairs, exclude=["b1"], prior=prior)
    h3, rep3 = PR.fit_hyper(pairs, exclude=["b2"], prior=prior, nu_mu=3)
    assert h3.nu_mu == 3 and h3.sigma_mu == PR.REFERENCE["sigma_mu"]
    h4, rep4 = PR.fit_hyper(pairs, exclude=["b2"], prior=prior, centre="zero")
    assert h4.mu0 == 0 and "mu0" not in rep4["estimated"] + rep4["fallback"]
    assert PR.build_prior(pairs, exclude=["b0", "b1", "b2"]) is None
    p = make_hier(back, hyper)([pairs[0].subject, pairs[0].responses[0].item], [])
    assert 0 < p < 1


def test_levels_and_variances_are_where_the_run_time_model_uses_them():
    """fit_hyper's level of a benchmark is at attribute score 0: its fitted mu
    plus the pool's mean standing less its mean attribute score. build_prior's
    variances add the standings' posterior variance to their spread, since
    they are posterior means (Laplace-EM's own fixed point for s2_t)."""
    pairs = make_pairs(6, 3, 100)
    for p in pairs:
        p.subject["provider"] = "openai" if p.subject_id < "s3" else "anthropic"
        p.subject["release_date"] = f"2025-0{1 + int(p.subject_id[1:])}-01"
    prior = PR.build_prior(pairs)
    hyper, rep = PR.fit_hyper(pairs, prior=prior)
    for b, ps in PR._by_benchmark(pairs).items():
        f = PR.benchmark_fit(ps)
        attr = np.mean([prior.attribute(p.subject)[0] for p in ps])
        assert rep["levels"][b] == pytest.approx(f["mu"] + np.mean(f["t"]) - attr, abs=1e-12)
        assert rep["levels_fitted"][b] == f["mu"]
    assert hyper.mu0 == pytest.approx(np.mean(list(rep["levels"].values())))
    rows = PR.standings(pairs)
    z, noise = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
    assert prior.meta["total_var"] == pytest.approx(np.mean(z ** 2) + noise.mean())
    for b, ps in PR._by_benchmark(pairs).items():
        f = PR.benchmark_fit(ps)
        t = np.array(f["t"])
        assert f["s2_t"] == pytest.approx(np.mean(t ** 2) + np.mean(f["t_var"]), rel=1e-3)


def test_benchmark_fits_are_cached_on_content():
    """Two pairs trading a success and a failure between items keep every count
    and must still get a fresh fit (a count key served the stale one)."""
    from paiec.evaluator import Response
    pairs = [p for p in make_pairs(4, 1, 60)]
    f1 = PR.benchmark_fit(pairs)
    p = pairs[0]
    labels = [r.label for r in p.responses]
    i1, i0 = labels.index(1), labels.index(0)
    swapped = list(p.responses)
    for i in (i1, i0):
        r = swapped[i]
        swapped[i] = Response(r.item_key, r.item, 1 - r.label)
    pairs[0] = type(p)(p.subject, p.subject_id, p.benchmark_id, swapped)
    f2 = PR.benchmark_fit(pairs)
    assert f2 is not f1 and f2["n_cells"] == f1["n_cells"] and f2["t"] != f1["t"]
    assert PR.benchmark_fit(pairs) is f2
    assert PR.benchmark_fit(pairs[::-1]) is not f2          # t follows the pairs' order


def test_nothing_left_means_reference_values_not_fitted_defaults():
    """Leaving every benchmark out must not hand back numbers fitted on them:
    each field takes the data-free REFERENCE and says so."""
    pairs = make_pairs(4, 2, 100)
    names = sorted({p.benchmark_id for p in pairs})
    hyper, rep = PR.fit_hyper(pairs, exclude=names)
    assert rep["estimated"] == [] and set(rep["fallback"]) == set(PR.REFERENCE)
    assert {k: getattr(hyper, k) for k in PR.REFERENCE} == PR.REFERENCE
    assert hyper.mu0 == 0.0 != Hyper().mu0 and hyper.included == ()
    # the pair's own width is on the Rasch scale, like the item's
    assert PR.REFERENCE["sigma_delta"] == PR.REFERENCE["sigma_d"] == 2.5
    s = subject("new")
    assert HierPredictor(None, hyper, floor=False, slip=False).predict([s, item(0, "b9")], []) \
        == pytest.approx(0.5, abs=1e-12)


def test_canonical_name_links_variants():
    assert PR.canon_name(subject("OpenAI o4-mini (high)")) == "o4 mini"
    assert PR.canon_name(subject("", subject_features_extra="model_config=x;source_model_name=Kimi K2")) \
        == "kimi k2"
    assert PR.canon_name({"normalized_name": None}) == ""
    assert PR.canon_name(subject("Gizmo (v2 (think)) pro")) == "gizmo pro"
    for bad in ("(" * 20_000, "(a" * 10_000 + ")" * 10_000, "x" * 200_000):
        t0 = time.perf_counter()
        PR.canon_name(subject(bad))
        PR.canon_name(subject("", subject_features_extra="source_model_name=" + bad))
        assert time.perf_counter() - t0 < 0.1
    model = HierPredictor(None, Hyper())
    a, b = subject("o4-mini", reasoning_effort="low"), subject("o4-mini", reasoning_effort="high")
    assert model.identity_key(b"a", a) == model.identity_key(b"b", b)
    full = HierPredictor(None, Hyper(), subject_key="full")
    assert full.identity_key(b"a", a) != full.identity_key(b"b", b)


# --- the identity components --------------------------------------------------------

def drawn_standings(seed, tau2, sd, n_b=4, n_names=10, extra=4, n_meta=0, effect=2.0, v=0.1):
    """Standings rows as prior.standings gives them, with a known shared part:
    z = f(provider, release date) + theta_name + delta, centred within each
    benchmark; n_names models on every benchmark, `extra` others on one each.
    With n_meta, that many models of provider meta are on b0 and b1 only, and
    one more on each of the two: meta's 2 n_meta + 2 rows reach Spec's 8 with
    both benchmarks in, so the full design has the column (effect `effect`)
    and a spec rebuilt without b0 or b1 would drop it."""
    rng = np.random.default_rng(seed)
    n = n_names + n_b * extra
    m = n + (n_meta + 2 if n_meta else 0)
    prov, days = rng.integers(0, 3, m), rng.integers(0, 900, m)
    prov[n:] = 3
    f = 0.8 * (prov == 0) - 0.5 * (prov == 2) + effect * (prov == 3) + 1.2 * (days - 450) / 450
    theta = rng.normal(0, math.sqrt(tau2), m)
    subs = [subject(f"model {k}", provider=("openai", "anthropic", "google", "meta")[prov[k]],
                    release_date=str(np.datetime64("2023-01-01") + int(days[k]))) for k in range(m)]
    rows = []
    for b in range(n_b):
        who = list(range(n_names)) + list(range(n_names + b * extra, n_names + (b + 1) * extra))
        if n_meta and b < 2:
            who += list(range(n, n + n_meta)) + [n + n_meta + b]
        z = f[who] + theta[who] + rng.normal(0, sd, len(who))
        rows += [(f"b{b}", subs[k], float(zk), v) for k, zk in zip(who, z - z.mean())]
    return rows


def test_identity_residuals_leave_the_benchmark_and_the_name_out():
    """A name's residual on b comes from a ridge fitted without b and without
    any row of the name, centred on that ridge's own mean over b, so it does
    not move with the name's standings elsewhere; a name seen once keeps the
    benchmark-out ridge. Every ridge is fitted on the full ridge's design,
    including a provider column a spec rebuilt on the training rows would
    drop. The smoother weights reproduce the centred prediction, at the
    run-time alpha and at the identity pass's."""
    rows = drawn_standings(0, 0.3, 0.7, n_b=3, n_names=5, extra=3, n_meta=3)
    z = np.array([r[2] for r in rows])
    names = [PR.canon_name(r[1]) for r in rows]
    bench = [r[0] for r in rows]
    _, spec, X, _ = PR._ridge(rows, 2.0)
    assert "meta" in spec.providers
    assert "meta" not in PR._ridge([r for r in rows if r[0] != "b0"], 2.0)[1].providers

    def by_hand(i, drop_name, alpha):
        tr = [j for j in range(len(rows)) if bench[j] != bench[i]
              and not (drop_name and names[j] == names[i])]
        te = [j for j in range(len(rows)) if bench[j] == bench[i]]
        c = np.linalg.solve(X[tr].T @ X[tr] + alpha * np.eye(X.shape[1]), X[tr].T @ z[tr])
        pred = X[te] @ c
        return z[i] - (pred[te.index(i)] - pred.mean()), pred[te.index(i)] - pred.mean()
    linked = names.index("model 1", 5)                  # model 1 on b1
    alone = names.index("model 6")                      # an extra on b0
    meta = names.index("model 14")                      # meta, on b0 and b1
    for alpha in (2.0, PR.ID_ALPHA):
        resid, weights = PR._residuals(rows, alpha)
        for i, drop in ((linked, True), (meta, True), (alone, False)):
            want, centred = by_hand(i, drop, alpha)
            assert resid[i] == pytest.approx(want, abs=1e-8)
            if drop:
                assert weights[i] @ z == pytest.approx(centred, abs=1e-8)
    resid, weights = PR._residuals(rows, 2.0)
    assert set(weights) == {i for i in range(len(rows))
                            if names[i] in {f"model {k}" for k in (0, 1, 2, 3, 4, 14, 15, 16)}}
    moved = list(rows)
    for j in range(len(rows)):
        if names[j] == "model 1" and bench[j] != "b1":
            moved[j] = (*rows[j][:2], rows[j][2] + 3.0, rows[j][3])
    again, _ = PR._residuals(moved, 2.0)
    assert again[linked] == pytest.approx(resid[linked], abs=1e-12)
    assert again[alone] != pytest.approx(resid[alone], abs=1e-6)
    assert np.all(np.isnan(PR._residuals([r for r in rows if r[0] == "b0"], 2.0)[0]))


def test_identity_tau2_is_recovered_on_standings_with_a_known_shared_part():
    """The shared part tau2 comes back within noise, for 0 and for 0.3, from
    standings drawn with it on a design where meta's column, worth 2 logits,
    has fewer than 8 rows once b0 or b1 is left out: over 200 sets the fixed
    estimator gave -0.003 +- 0.003 and 0.281 +- 0.010 (a little low at 0.3:
    the offsets and the mean are taken from the same cells). Before the fix
    (a spec rebuilt on each training subset, residuals at alpha 2) these 30
    sets gave +0.041 +- 0.008 for 0, which fails here (step-2b
    fix_id/test_size.txt); on standings drawn on the public design, +0.144.
    What is left is the bias the ridges share (paiec.prior.from_standings),
    about +0.02 there."""
    for tau2 in (0.0, 0.3):
        got = []
        for seed in range(30):
            prior = PR.from_standings(drawn_standings(seed, tau2, 0.7, n_names=20, extra=8, n_meta=3))
            got.append(prior.meta["tau2_res"])
            assert prior.meta["linked"] == 23 and "meta" in prior.spec.providers
            assert 0 < prior.meta["link_noise"] < 0.3
        got = np.array(got)
        se = got.std(ddof=1) / math.sqrt(len(got))
        assert se < (0.015 if tau2 == 0 else 0.04), (tau2, se)
        assert abs(got.mean() - tau2) < 3 * se, (tau2, got.mean(), se)


# --- hooks for moving the priors ------------------------------------------------------

def _theta_prior_before_hooks(self, subject):
    """HierPredictor._theta_prior as it was before Hyper.attr_scale and
    Hyper.shift (commit bba726c)."""
    h, cfg, pr = self.hyper, self.cfg, self.prior
    base = (0.0, h.sigma_theta ** 2 + h.sigma_attr ** 2)
    try:
        subject = {f: H._text(subject.get(f)) for f in H.SUBJECT_FIELDS}
        m, V = 0.0, h.sigma_theta ** 2
        attr = cfg.attributes and pr is not None and pr.has_attributes
        if attr:
            ma, u = pr.attribute(subject)
            m, V = m + ma, V + u
        else:
            V += h.sigma_attr ** 2
        if cfg.identity and pr is not None:
            obs = pr.identity(subject, residual=attr)
            if obs is not None:
                rbar, vid = obs
                w = min(V / (V + vid), h.id_cap)
                m, V = m + w * rbar, (1 - w) ** 2 * V + w * w * vid
        if math.isfinite(m) and math.isfinite(V) and V > 0:
            return m, V
    except Exception:
        pass
    return base


def hooks_run():
    """Subjects with attributes, one in the toy identity table, on two
    benchmarks with features; targets on labeled and new items, a new
    benchmark and a new subject."""
    subs = [subject("Gizmo-2 (high)", provider="OpenAI", release_date="2025-06-01"),
            subject("widget", provider="anthropic", release_date="2024-03-01", reasoning_effort="high"),
            subject("plain")]
    labeled = [[[s, item(j, bid, f"tier=t{j % 3}")], int((j + k + b) % 3 == 0)]
               for b, bid in enumerate(("benchmark_1", "benchmark_2"))
               for k, s in enumerate(subs) for j in range(6 + 3 * k)]
    targets = [[s, item(j, bid, f"tier=t{j % 4}")]
               for s in subs + [subject("newcomer", provider="openai", release_date="2025-01-01")]
               for bid in ("benchmark_1", "benchmark_2", "benchmark_3") for j in (0, 40)]
    return labeled, targets


@pytest.mark.parametrize("hyper", [Hyper(), Hyper(sigma_mu=1.829, nu_mu=3.0)])
def test_the_prior_hooks_at_their_defaults_change_nothing(hyper):
    """attr_scale 1 and shift 0 give bit-for-bit what the code gave before
    they existed (the old _theta_prior, the only reader of either), under
    every flag that changes how theta's prior is used, with and without a
    subject prior; moved, they do move the predictions."""
    from dataclasses import replace
    import types
    labeled, targets = hooks_run()
    assert (hyper.attr_scale, hyper.shift) == (1.0, 0.0)
    for prior in (toy_prior(), None):
        for flags in ({}, {"attributes": False}, {"identity": False}, {"link": False},
                      {"delta": False}, {"line": False}):
            new = HierPredictor(prior, hyper, **flags)
            old = HierPredictor(prior, hyper, **flags)
            old._theta_prior = types.MethodType(_theta_prior_before_hooks, old)
            for lab in ([], labeled[:7], labeled):
                assert [new.predict(t, lab) for t in targets] == \
                    [old.predict(t, lab) for t in targets], (prior is None, flags, len(lab))
            assert new.failures == old.failures == 0
    base = [HierPredictor(toy_prior(), hyper).predict(t, labeled) for t in targets]
    for moved in (replace(hyper, attr_scale=0.5), replace(hyper, shift=-0.4)):
        ps = [HierPredictor(toy_prior(), moved).predict(t, labeled) for t in targets]
        assert max(abs(a - b) for a, b in zip(ps, base)) > 0.01


def test_attr_scale_scales_the_attribute_mean_and_shift_adds_to_it():
    """theta's prior mean is shift + attr_scale * m_s + the identity term, whose
    weight depends on the variance alone; the variance does not move. Without
    attributes the scale does nothing and the shift still applies, also to
    the no-attribute fallback of a prior that fails."""
    from dataclasses import replace
    prior, h = toy_prior(), Hyper()
    plain = subject("widget", provider="anthropic", release_date="2024-03-01")
    known = subject("Gizmo-2 (high)", provider="OpenAI", release_date="2025-06-01")
    for scale, shift in ((0.0, 0.0), (0.5, 0.0), (2.0, -0.7)):
        hh = replace(h, attr_scale=scale, shift=shift)
        ma, u = prior.attribute(plain)
        mt, vt = HierPredictor(prior, hh).theta_prior(b"p", plain)
        assert mt == pytest.approx(shift + scale * ma, abs=1e-12)
        assert vt == pytest.approx(h.sigma_theta ** 2 + u, abs=1e-12)
        m0, v0 = HierPredictor(prior, h).theta_prior(b"k", known)
        mt, vt = HierPredictor(prior, hh).theta_prior(b"k", known)
        ma, _ = prior.attribute(known)
        assert mt - shift - scale * ma == pytest.approx(m0 - ma, abs=1e-12) and vt == v0
        off = HierPredictor(prior, hh, attributes=False, identity=False)
        assert off.theta_prior(b"p", plain) == pytest.approx((shift, h.sigma_theta ** 2 + h.sigma_attr ** 2))
    broken = toy_prior()
    broken.coef = broken.coef[:-1]                      # attribute() raises
    got = HierPredictor(broken, replace(h, shift=0.3)).theta_prior(b"p", plain)
    assert got == pytest.approx((0.3, h.sigma_theta ** 2 + h.sigma_attr ** 2))


def test_shift_is_the_level_moved_by_the_same_amount():
    """Every eta holds one level and one theta, so moving every subject's prior
    by c is the same model as moving mu0 by c: the same predictions at every
    budget, Gaussian or Student-t level, up to the fit's tolerance."""
    from dataclasses import replace
    labeled, targets = hooks_run()
    for h in (Hyper(), Hyper(sigma_mu=1.829, nu_mu=3.0)):
        for c in (-1.5, 0.8):
            a = HierPredictor(toy_prior(), replace(h, shift=c))
            b = HierPredictor(toy_prior(), replace(h, mu0=h.mu0 + c))
            for lab in ([], labeled[:7], labeled):
                pa, pb = [a.predict(t, lab) for t in targets], [b.predict(t, lab) for t in targets]
                assert pa == pytest.approx(pb, abs=1e-9), (h.nu_mu, c, len(lab))


def test_the_level_prior_is_overridden_by_replace():
    """mu0, sigma_mu and nu_mu set by dataclasses.replace are what a new
    benchmark's B0 prediction uses, and the hooks survive prior.json."""
    from dataclasses import replace
    prior = toy_prior()
    s, it = subject("widget", provider="anthropic"), item(0, "benchmark_9")
    for h in (replace(Hyper(), mu0=-3.0, sigma_mu=0.7), replace(Hyper(), mu0=1.0, sigma_mu=4.0)):
        model = HierPredictor(prior, h, floor=False)
        mt, vt = model.theta_prior(b"any", s)
        v = h.sigma_mu ** 2 + vt + h.sigma_delta ** 2 + h.sigma_d ** 2 + h.sigma_g ** 2
        want = (1 - h.slip) * float(sig(h.mu0 + mt + math.sqrt(v) * ZF) @ WF)
        assert model.predict([s, it], []) == pytest.approx(want, abs=1e-9)
    t3 = replace(Hyper(), nu_mu=3.0)
    assert HierPredictor(prior, t3).predict([s, it], []) != HierPredictor(prior, Hyper()).predict([s, it], [])
    moved = replace(Hyper(), attr_scale=0.5, shift=-0.7, mu0=-2.0, sigma_mu=1.5, nu_mu=4.0)
    assert PR.from_json(json.loads(json.dumps(PR.to_json(prior, moved))))[1] == moved
    old = {k: v for k, v in Hyper().to_dict().items() if k not in ("attr_scale", "shift")}
    assert Hyper.from_dict(old) == Hyper()               # a prior.json from before the hooks
    with pytest.raises(ValueError):
        PR.from_json({**PR.to_json(None, Hyper()), "hyper": {**Hyper().to_dict(), "attr_scale": -1.0}})
