"""experiments/dense_b1_check.py (S1 stage 1) on synthetic data (no download needed).

The exact posterior predictive is checked where it can be computed another way: one
subject on one benchmark with no group keys, where every label's eta is mu + theta +
delta - e_i, so the posterior of x reaches every prediction through the one sum
s = mu + theta + delta, whose prior is Gaussian. Then E[sigmoid(eta_t)] is a
one-dimensional integral over s of the prior times each labeled item's likelihood
(itself an integral over its residual), computed here by direct quadrature on fine
grids, independently of the script's code: for a target on a new item, on a labeled
item (its residual read given s), and with floored (multiple-choice) successes among
the labels and a floored target. Defensive importance sampling (the estimate) must
reach it within four of its own SEs, and the SMC sampler (the second estimate) within
3e-3; hier's line read, exact for one pair, within 2e-3. Also: the delta-method SE of
a Brier difference against its spread over independent runs, the log posterior
against hier's Problem.state and a finer grid, the tabulated Gaussian convolution,
the Pareto k-hat on weights of known tail, target picking, the bootstrap and the
gate's reading."""
import math

import numpy as np
import pytest
from scipy.special import expit, log_expit

from experiments import dense_b1_check as S
from paiec.hier import HierPredictor, Hyper
from tests.synth import make_pair

MCQ = "Which one?\n(A) 1\n(B) 2\n(C) 3\n(D) 4"


def labeled_list(pair, items, labels, mcq=()):
    """[[subject, item], y] for these item indices, items in `mcq` given four options."""
    subject = {k: v for k, v in pair.subject.items() if not k.startswith("_")}
    by = {r.item_key: r.item for r in pair.responses}
    out, inputs = [], {}
    for i in sorted(set(items) | set(mcq)):
        it = {k: v for k, v in by[f"b0:{i}"].items() if not k.startswith("_")}
        if i in mcq:
            it["item_content"] = f"{MCQ}\n(item {i})"
        inputs[i] = [subject, it]
    for i, y in zip(items, labels):
        out.append([[dict(inputs[i][0]), dict(inputs[i][1])], int(y)])
    return out, inputs


def reference(fit, tg_item, target_v, c_t, slip):
    """E over the posterior of s of E[sigmoid(eta_t) | s], by quadrature: s on 6,001
    nodes over the prior's +-12 sds, each item residual on 4,001 nodes over +-12
    prior sds. tg_item: None (new item) or the labeled item the target reads."""
    prob = fit.prob
    M, W = float(prob.m.sum()), float(prob.V.sum())
    s = M + math.sqrt(W) * np.linspace(-12, 12, 6001)
    logp = -0.5 * (s - M) ** 2 / W
    lq = {}
    for j in range(prob.n_items):
        sd = math.sqrt(prob.s2e[j])
        e = sd * np.linspace(-12, 12, 4001)
        lw = -0.5 * e ** 2 / prob.s2e[j]
        tot = np.zeros((len(s), len(e)))
        for o in np.flatnonzero(prob.item == j):
            z = s[:, None] + prob.off[o] - e[None, :]
            y, c = prob.y[o], prob.c[o]
            if c > 0 and y > 0.5:
                tot += np.log(c + (1 - c) * expit(z))
            elif c > 0:
                tot += np.log1p(-c) + log_expit(-z)
            else:
                tot += log_expit(z if y > 0.5 else -z)
        L = lw[None, :] + tot
        mx = L.max(1)
        logp = logp + mx + np.log(np.exp(L - mx[:, None]).sum(1))
        lq[j] = (e, np.exp(L - mx[:, None]))
    w = np.exp(logp - logp.max())
    w /= w.sum()
    if tg_item is None:
        z = np.linspace(-12, 12, 4001)
        g = np.exp(-z ** 2 / 2)
        g /= g.sum()
        F = expit(s[:, None] + math.sqrt(target_v) * z[None, :]) @ g
    else:
        e, q = lq[tg_item]
        q = q / q.sum(1, keepdims=True)
        F = (q * expit(s[:, None] - e[None, :])).sum(1) if target_v == 0 else \
            (q * S.hexact((s[:, None] - e[None, :]).ravel(), target_v).reshape(q.shape)).sum(1)
    return c_t + (1 - c_t - slip) * float(w @ F)


def one_pair_case(labels, items, mcq=()):
    pair = make_pair("s0", "b0", n_items=40, seed=5)
    labeled, inputs = labeled_list(pair, items, labels, mcq)
    model = HierPredictor(None, Hyper())
    fit = model.fit_for(labeled)
    return pair, labeled, inputs, model, fit


def check_reduction(fit, tg):
    """The one-dimensional reduction holds: every label and the target load mu,
    theta and delta with coefficient 1 and nothing else."""
    prob = fit.prob
    assert prob.p == 3
    assert np.all(prob.vals == 1.0) and np.all(prob.off == 0.0)
    assert sorted(tg.cols.tolist()) == [0, 1, 2] and np.all(tg.coef == 1.0)


@pytest.mark.parametrize("case", ["new item", "labeled item", "floored labels"])
def test_exact_matches_direct_quadrature(case):
    items = [0, 1, 2, 3, 4, 5]
    labels = [1, 0, 0, 1, 0, 0]
    # the floored items carry successes (a floored failure only adds a constant) and
    # the floored case's target is a four-option item too, so its floor enters the
    # prediction
    mcq = (0, 3, 20) if case == "floored labels" else ()
    pair, labeled, inputs, model, fit = one_pair_case(labels, items, mcq)
    tgt_i = 3 if case == "labeled item" else 20
    if tgt_i not in inputs:
        _, extra = labeled_list(pair, [tgt_i], [0])
        inputs[tgt_i] = extra[tgt_i]
    inp = inputs[tgt_i]
    ex = S.Exact(fit)
    t = S.Target(fit, model, inp)
    check_reduction(fit, t)
    assert t.kind == ("own" if case == "labeled item" else "new")
    if case == "floored labels":
        assert int(((fit.prob.c > 0) & (fit.prob.y > 0.5)).sum()) == 2 and t.c > 0
    j = t.items[0] if t.items else None
    ref = reference(fit, j, t.v, t.c, t.slip)

    # the estimate: defensive importance sampling, within four of its own SEs
    isr = S.defensive_is(ex, [t], np.random.default_rng(11), n=40000, batches=4)
    p_is = t.p(isr["F"][0])
    se = (1 - t.c - t.slip) * isr["se"][0]
    assert 0 < se < 2e-3
    assert abs(p_is - ref) < 4 * se + 2e-4, (case, p_is, ref, se)
    assert isr["batches"].shape == (4, 1)
    # the second estimate: SMC
    reps = [S.smc(ex, [t], np.random.default_rng([7, k]), n=2000, final_moves=20)["F"][0]
            for k in range(3)]
    p_smc = t.p(np.mean(reps))
    assert abs(p_smc - ref) < 3e-3, (case, p_smc, ref)
    # the prior alone would be far off: the labels move the prediction
    assert abs(model.predict(inp, []) - ref) > 5e-3
    # for one pair hier's line read is the exact marginal (paiec/hier.py): within 2e-3
    # here (1e-5 when this test was written), where the Laplace Gaussian alone is not
    assert abs(model.predict(inp, labeled) - ref) < 2e-3
    if case != "labeled item":
        lap = HierPredictor(None, model.hyper, line=False).predict(inp, labeled)
        assert abs(lap - ref) > 3e-3


def test_delta_se_of_a_brier_difference_matches_the_spread_over_runs():
    pair, labeled, inputs, model, fit = one_pair_case([1, 0, 0, 1, 0, 0], [0, 1, 2, 3, 4, 5])
    _, extra = labeled_list(pair, [20, 21, 3], [0, 0, 0])
    tgs = [S.Target(fit, model, extra[i]) for i in (20, 21, 3)]
    ys = [1, 0, 0]
    ex = S.Exact(fit)
    ph = np.array([0.3, 0.3, 0.6])
    D, SE = [], []
    for k in range(10):
        isr = S.defensive_is(ex, tgs, np.random.default_rng(100 + k), n=4000, batches=2)
        pe = np.array([t.p(f) for t, f in zip(tgs, isr["F"])])
        D.append(np.mean((ph - ys) ** 2 - (pe - ys) ** 2))
        SE.append(S.delta_se_diff(isr, tgs, ys))
    sd = np.std(D, ddof=1)
    assert 0.5 < np.mean(SE) / sd < 2.0, (np.mean(SE), sd)


def test_logpost_matches_hier_and_a_finer_grid():
    pair, labeled, inputs, model, fit = one_pair_case([1, 0, 1, 1, 0], [0, 1, 2, 3, 3], mcq=(1,))
    ex = S.Exact(fit)
    x0, L = S.laplace(fit)
    X = np.vstack([x0, x0 + np.random.default_rng(0).standard_normal((20, ex.p)) @ L.T])
    mine, _ = ex.logpost(X)
    theirs = np.array([fit.prob.state(x).lp for x in X])
    fine, _ = S.Exact(fit, refine=3).logpost(X)
    assert np.max(np.abs(mine - theirs)) < 1e-3
    assert np.max(np.abs(mine - fine)) < 1e-8
    # the residual's node weights given x are a distribution
    _, Q = ex.logpost(X, keep=[0, 2])
    assert Q.shape == (len(X), 2, ex.K)
    assert np.allclose(Q.sum(2), 1.0)


def test_htable():
    z = np.linspace(-40, 40, 3001)
    for v in (0.0, 0.3, 7.0, 40.0):
        assert np.max(np.abs(S.HTable(v)(z) - S.hexact(z, v))) < 1e-6
    # v = 0 is the sigmoid; a large variance flattens toward 0.5
    assert np.allclose(S.HTable(0.0)(z), expit(z))
    assert abs(float(S.HTable(400.0)(np.array([3.0]))[0]) - 0.5) < 0.1


def test_khat_on_known_tails():
    rng = np.random.default_rng(0)
    assert S.khat(rng.normal(0, 0.3, 20000)) < 0.3
    k = S.khat(np.log(rng.random(20000) ** -0.5))
    assert 0.35 < k < 0.7


class _Slot:
    def __init__(self, sid, n):
        self.subject_id, self.benchmark_id = sid, "b"
        self.targets = [([{"s": sid}, {"i": i}], i % 2, f"{sid}:{i // 2}") for i in range(n)]


def test_pick_targets_round_robin_and_deterministic():
    slots = [_Slot(f"s{k}", n) for k, n in enumerate((3, 40, 40, 7))]
    picks = S.pick_targets(slots, n=20)
    assert picks == S.pick_targets(slots, n=20)
    per = np.bincount([p[0] for p in picks], minlength=4)
    assert per.sum() == 20
    assert per[0] == 3                              # all it has
    assert max(per[1:]) - min(per[1:]) <= 1         # the rest share equally
    # repeated responses of one input are distinct targets
    assert len({(p[0], p[3]) for p in picks}) == 20


def test_bootstrap_and_gate_reading():
    rng = np.random.default_rng(1)
    tb, cb = S.boot(np.full(50, 0.004), np.repeat(np.arange(10), 5), rng, n=500)
    assert tb[:2] == (0.004, 0.004) and cb[:2] == (0.004, 0.004)

    def summ(mean, lo, hi):
        return {S.ck_key(S.GATE_BENCH, S.GATE_SCOPE, S.GATE_BUDGET):
                {"hier - exact": {"mean": mean, "cluster_boot": (lo, hi, 0.001),
                                  "target_boot": (lo, hi, 0.001), "mc_se": 1e-4}}}
    assert S.gate(summ(0.004, 0.001, 0.007))["verdict"] == "stage 2 opens"
    assert S.gate(summ(0.004, -0.001, 0.009))["verdict"] == "S1 closes"     # interval holds 0
    assert S.gate(summ(0.002, 0.001, 0.003))["verdict"] == "S1 closes"      # below 0.003
    assert S.gate(summ(-0.004, -0.007, -0.001))["verdict"] == "S1 closes"   # excludes 0, wrong side
    assert S.gate({})["verdict"] is None


def test_design_constants_match_the_plan():
    assert S.LEVEL == S.PD.model_level()
    assert S.GATE_MIN == 0.003 and S.GATE_BUDGET == 1 and S.GATE_BENCH == "real_webagents"
    assert 50 <= S.N_TARGETS <= 100
    assert S.CHECK_BUDGETS == (1, 3)
