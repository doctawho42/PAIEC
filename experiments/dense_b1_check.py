"""S1 stage 1: hier's prediction against the exact posterior predictive of its own model
at dense B1 and B3 (docs/plans/p2_final_plan.md, section 3, check S1, gate G1).

What it asks. On dense real_webagents with per-pair splits, hier's pooling costs
+0.015 +- 0.010 of Brier at B1 (docs/findings.md, "Pooling under the verified
protocol", "Not explained"; experiments/pooling_decomposition.py, ship minus ship_own).
A cost at B1 is the signature CLAUDE.md says to check first for a mishandled
second-order term. This script computes, for sampled targets, the posterior
predictive that hier's model implies when nothing is approximated, and compares it
with hier's prediction as shipped (Laplace fit plus the line read). If the two
agree, the cost is the model's (pooling under misspecification), not the
approximation's, and S1 closes.

The gate, as the plan fixes it (quoted, binding):

    G1, stage 2 only if Brier(hier) - Brier(exact) at B1 on real_webagents is at
    least 0.003 and its 95% interval excludes 0; otherwise S1 closes and the result
    goes into §7 and App I.1.

How the gate is read here (fixed before any number was computed): the difference is
the mean over the sampled targets of (p_hier - y)^2 - (p_exact - y)^2 at B1 on dense
real_webagents under split scope 'pair', the scope of the +0.015 that S1 diagnoses;
the 95% interval is the pair-cluster bootstrap percentile interval (BOOTS resamples
of the run's pairs). The target bootstrap interval, the Monte Carlo SE of the
difference and the same reading under scope 'benchmark' are reported beside it.

Design (DESIGN below):
  runs      official.dense_run of real_webagents (the gate) and of the controls
            matharena and multi_swebench; split seed 0; the platform's random
            acquisition policy; the checkpoints of experiments/pooling_decomposition.py
            (PD.checkpoints, its EVAL_CAP of 32 evaluation items per pair on the two
            controls, none on real_webagents), so `labeled` at every checkpoint is the
            full dense run's and the targets are a subsample of that study's.
  budgets   B1 and B3.
  scopes    'pair' (primary) and 'benchmark' (secondary).
  targets   N_TARGETS = 100 evaluation responses per benchmark and scope: pairs in
            digest order, round robin, each pair's responses in digest order (one
            stable_hash per response), so allocation over pairs is equal to within one,
            as in the pair-averaged official Brier. The same targets at B1 and B3.
  prior     leave-one-benchmark-out, as every experiment that scores a public
            benchmark: level_calibration.bundle(benchmark, 0.0), i.e.
            paiec.prior.build on every public pair outside the benchmark, with
            submission/model.py's LEVEL (checked) written over the hyperparameters;
            exactly pooling_decomposition's 'ship'. Not the shipped prior.json: it is
            fitted on all five public benchmarks, so its identity table holds these
            very subjects' standings on the scored benchmark, and the +0.015 under
            diagnosis was measured with the leave-one-out bundle. Both hier and the
            exact predictive use the same prior and hyperparameters, so the comparison
            isolates the inference.
  hier      HierPredictor(prior, hyper).predict, a fresh instance per checkpoint as on
            the platform: the shipped configuration, line read on. Also, as
            diagnostics on the same targets: line=False (the Laplace Gaussian alone)
            and ship_own (pooling_decomposition.OwnOnly, the target pair's own labels).
  exact     the posterior predictive under hier's own model: the same likelihood
            (logistic with the MCQ floor, item residual integrated), priors and
            hyperparameters, taken from the fit hier makes of the same `labeled`
            (_Fit: design columns, prior means and variances, floors, item variances;
            _Fit._target: the target's coefficients on x, the prior of what no label
            touched, its own labeled item or folded carriers). Nothing hier
            approximates is reused: the collapsed log posterior of x is recomputed
            here with every item residual integrated on its own fixed trapezoid grid
            (Exact.logpost; checked against hier's Problem.state, and against a grid
            three times finer), and E[sigmoid(eta_t)] is
                integral over the joint posterior of (x, e_j) of sigmoid(eta_t),
            by importance sampling over the full joint posterior of x. Given x, the
            target's own labeled item's residual (or a folded carrier's) is integrated
            exactly on its grid and the untouched Gaussian part by a tabulated
            convolution (HTable), so each draw carries E[sigmoid(eta_t) | x] (Target.F).
            The estimate (defensive_is): IS_DRAWS draws in IS_BATCHES independent
            batches from the defensive mixture q = (1 - IS_ALPHA) N(x0, S) + IS_ALPHA
            x0 + L t, where N(x0, S) is the Laplace Gaussian hier fits (mode x0, S =
            LL') and t has independent Student-t coordinates with IS_DF degrees of
            freedom (Hesterberg 1995). Every label's likelihood is at most 1, so the
            posterior has the prior's Gaussian tails at most, and the t component's
            polynomial tails bound the weights pi / q in every direction: the variance
            is finite whatever the posterior's skew, which the Laplace Gaussian alone
            does not guarantee (a floored success or a run of failures leaves a tail
            as wide as the prior's; tests/test_dense_b1_check.py has such a case). Its
            Monte Carlo SE: the delta method for every prediction and every Brier
            difference (the difference is linear in the exact predictions to first
            order, with the same coefficients whichever predictor it is taken
            against), and the spread over the batches beside it. A second estimate,
            by another algorithm (smc): a sequential Monte Carlo sampler (adaptive
            tempering from the Laplace Gaussian to the posterior, systematic
            resampling, Metropolis moves reversible with respect to that Gaussian;
            Del Moral, Doucet and Jasra 2006), REPLICATES independent runs of
            PARTICLES particles, compared target by target in units of the combined
            SE.

Statistics. Per benchmark, scope and budget: Brier of each predictor on the targets;
the difference hier - exact (and laplace - exact, own - exact) with a pair-cluster and
a target bootstrap percentile interval (BOOTS resamples) and the Monte Carlo SE;
mean |p_hier - p_exact| and the mean shift of hier toward 0.5; the population Brier of
hier over every scored target of the checkpoint (reproducing
pooling_decomposition's stored per-pair Brier of 'ship' bit for bit where its rows are
on disk) beside the sample's.

Checks (recorded in the results file):
  - LEVEL equals submission/model.py's
  - hier on every scored target reproduces pooling_decomposition's stored per-pair
    Brier of 'ship' at B1 and B3 (data/pooling_decomposition_rows/dense, when present)
  - Exact.logpost against hier's Problem.state(x).lp at the mode and at Laplace
    draws, and against a grid of GRID_CHECK times the nodes
  - the SMC estimate against the importance-sampling one, per target, in units of
    their combined SE; the importance weights' ESS, Pareto k-hat and largest weight
  - tests/test_dense_b1_check.py: the exact predictor on a one-pair case reduced to a
    one-dimensional integral computed by direct quadrature

Run:
  OPENBLAS_NUM_THREADS=1 python experiments/dense_b1_check.py score [--only real_webagents.pair]
  python experiments/dense_b1_check.py status
  python experiments/dense_b1_check.py summarise

Output: results/dense_b1_check.json (OUT): design, per checkpoint the target rows
(public subject id, a sha256 digest of the official input, label, every prediction,
the Monte Carlo SE) and diagnostics, the summary, the gate and the provenance. No item
text is written. The score stage writes each checkpoint into OUT as it finishes and
skips checkpoints already there (resumable).
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import expit, log_expit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import experiments.level_calibration as LC  # noqa: E402
import experiments.pooling_decomposition as PD  # noqa: E402
from paiec import data as D  # noqa: E402
from paiec.evaluator import BUDGETS, stable_hash  # noqa: E402
from paiec.hier import HierPredictor, _Fit  # noqa: E402
from paiec.predict import _probability, item_key  # noqa: E402

OUT = os.path.join(ROOT, "results", "dense_b1_check.json")
PD_ROWS = os.path.join(ROOT, "data", "pooling_decomposition_rows", "dense")

# --- the design (fixed before any number was computed) --------------------------------------

LEVEL = {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}
GATE_BENCH, GATE_SCOPE, GATE_BUDGET = "real_webagents", "pair", 1
GATE_MIN = 0.003
GATE_TEXT = ("G1, stage 2 only if Brier(hier) - Brier(exact) at B1 on real_webagents is at "
             "least 0.003 and its 95% interval excludes 0; otherwise S1 closes and the result "
             "goes into §7 and App I.1.")
BENCHES = ("real_webagents", "matharena", "multi_swebench")
SCOPES = ("pair", "benchmark")
CHECK_BUDGETS = (1, 3)
N_TARGETS = 100
TARGET_SALT = "s1 dense b1 check targets"
SPLIT_SEED = 0
#: the estimate: draws from the defensive mixture, independent batches, the mixture
#: weight and degrees of freedom of its Student-t component (defensive_is)
IS_DRAWS = 200_000
IS_BATCHES = 10
IS_ALPHA = 0.1
IS_DF = 4.0
#: the second estimate: SMC particles per replicate, independent replicates, the ESS
#: fraction each tempering step keeps, Metropolis moves per tempering step and at the
#: posterior (every particle of every move at the posterior enters the estimate)
PARTICLES = 2000
REPLICATES = 4
TAU = 0.5
MOVES = 5
FINAL_MOVES = 25
RHO0 = 0.8
ACC_LO, ACC_HI = 0.2, 0.6
#: item-residual grid: U prior sds either side, node spacing at most H_MAX and at most
#: H_FRAC of the narrowest posterior sd the item's labels allow (logistic curvature 1/4)
GRID_U = 9.0
H_MAX = 0.8
H_FRAC = 0.9
GRID_CHECK = 3
#: the tabulated E[sigmoid(z + sqrt(v) Z)]: z range, step, trapezoid nodes in Z
H_ZMAX, H_STEP, H_NODES = 60.0, 0.005, 401
#: elements (particles x labels x nodes) per chunk of Exact.logpost
CHUNK = 3_000_000
BOOTS = 10000
BOOT_SEED = 20261007
SEED = 20261007
LIB_FILES = ("paiec/hier.py", "paiec/prior.py", "paiec/predict.py", "paiec/official.py",
             "paiec/subjects.py", "paiec/mcq.py", "paiec/fitting.py", "paiec/data.py",
             "paiec/evaluator.py", "paiec/testlike.py", "experiments/level_calibration.py",
             "experiments/pooling_decomposition.py", "submission/model.py")
DATA_TABLES = ("response.parquet", "items.parquet", "subjects.parquet")
PUBLIC = ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")


# --- helpers ------------------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def r(x, n=6):
    return None if x is None else round(float(x), n)


def utcnow():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
    os.replace(tmp, path)


def ck_key(bench, scope, budget):
    return f"{bench}.{scope}.B{budget}"


# --- the exact posterior --------------------------------------------------------------------------

class Exact:
    """The posterior of x under hier's model for one `labeled` list, without
    approximation, from the fit hier makes of it (`fit`, a paiec.hier._Fit): its
    Problem gives the design (cols, vals, off), labels, floors, item variances and the
    Gaussian prior (m, V) of every column; Student-t levels are not handled (the
    shipped level is Gaussian).

    logpost(X) is log p(x | labels) up to the constant hier's Problem.state drops: the
    Gaussian prior's exponent plus, for every labeled item, the log of the integral of
    N(e; 0, s2e_i) times its labels' likelihoods at eta_o(x) - e, each success's
    c + (1 - c) sigmoid and each failure's (1 - c) sigmoid(-.), by the trapezoid rule
    on the item's own grid (GRID_U prior sds either side; spacing set by the item's
    label count, see grid()). The trapezoid is exponentially accurate for these
    integrands; the check is a grid GRID_CHECK times finer."""

    def __init__(self, fit, refine=1):
        prob = fit.prob
        if prob is None:
            raise ValueError("nothing labeled: the posterior is the prior")
        if np.any(prob.nu > 0):
            raise NotImplementedError("Student-t levels are not handled")
        self.fit, self.prob = fit, prob
        self.p = prob.p
        order = np.argsort(prob.item, kind="stable")
        self.cols, self.vals = prob.cols[order], prob.vals[order]
        self.off, self.y, self.c = prob.off[order], prob.y[order], prob.c[order]
        self.item = prob.item[order]
        self.n_items = prob.n_items
        self.starts = np.searchsorted(self.item, np.arange(self.n_items))
        cnt = np.bincount(self.item, minlength=self.n_items)
        if np.any(cnt == 0):
            raise ValueError("an item without labels")
        self.s2e = np.asarray(prob.s2e, float)
        self.E, self.lw0 = grid(self.s2e, cnt, refine)
        self.K = self.E.shape[1]
        self.EL = self.E[self.item]
        self.sgn = np.where(self.y > 0.5, 1.0, -1.0)
        self.fs = np.flatnonzero((self.c > 0) & (self.y > 0.5))
        self.ff = np.flatnonzero((self.c > 0) & (self.y <= 0.5))
        self.logc_fs = np.log(self.c[self.fs])
        self.log1mc_fs = np.log1p(-self.c[self.fs])
        self.log1mc_ff = np.log1p(-self.c[self.ff])
        self.m, self.V = np.asarray(prob.m, float), np.asarray(prob.V, float)
        self.evals = 0

    def logpost(self, X, keep=None):
        """(lp, Q): lp per row of X, and for the items `keep` their residuals'
        posterior node weights given each row, (rows, len(keep), K) (None if keep
        is None)."""
        X = np.atleast_2d(np.asarray(X, float))
        n, L = len(X), len(self.y)
        step = max(1, CHUNK // max(1, L * self.K))
        lp = np.empty(n)
        Q = None if keep is None else np.empty((n, len(keep), self.K))
        for a in range(0, n, step):
            b = min(n, a + step)
            lp[a:b], q = self._chunk(X[a:b], keep)
            if keep is not None:
                Q[a:b] = q
        self.evals += n
        return lp, Q

    def _chunk(self, X, keep):
        eta = np.einsum("ls,nls->nl", self.vals, X[:, self.cols]) + self.off
        Z = eta[:, :, None] - self.EL[None, :, :]
        ll = log_expit(self.sgn[None, :, None] * Z)
        if len(self.fs):
            ll[:, self.fs] = np.logaddexp(self.logc_fs[None, :, None],
                                          self.log1mc_fs[None, :, None] + log_expit(Z[:, self.fs]))
        if len(self.ff):
            ll[:, self.ff] += self.log1mc_ff[None, :, None]
        F = np.add.reduceat(ll, self.starts, axis=1) + self.lw0[None]
        mx = F.max(2)
        logI = mx + np.log(np.exp(F - mx[:, :, None]).sum(2))
        r_ = X - self.m
        lp = logI.sum(1) - 0.5 * (r_ * r_ / self.V).sum(1)
        q = None
        if keep is not None:
            q = np.exp(F[:, keep, :] - logI[:, keep, None])
        return lp, q


def grid(s2e, cnt, refine=1):
    """(nodes E (items, K), log weights lw0 (items, K)): every item on its own uniform
    grid over [-GRID_U, GRID_U] prior sds, the log trapezoid weight plus the log
    N(e; 0, s2e) density; one K for all, the largest any item needs (spacing at most
    H_MAX, and at most H_FRAC times 1 / sqrt(n / 4 + 1 / s2e), the narrowest posterior
    sd n labels can give), times `refine`."""
    sd = np.sqrt(s2e)
    h = np.minimum(H_MAX, H_FRAC / np.sqrt(cnt / 4.0 + 1.0 / s2e))
    K = int(np.max(np.ceil(2 * GRID_U * sd / h))) + 1
    K = (K - 1) * int(refine) + 1
    u = np.linspace(-GRID_U, GRID_U, K)
    E = sd[:, None] * u[None, :]
    w = np.full(K, 1.0)
    w[0] = w[-1] = 0.5
    lw0 = (np.log(w)[None, :] + np.log(E[:, 1] - E[:, 0])[:, None]
           - E ** 2 / (2 * s2e[:, None]) - 0.5 * np.log(2 * math.pi * s2e)[:, None])
    return E, lw0


class HTable:
    """E[sigmoid(z + sqrt(v) Z)], Z standard normal, tabulated in z for one v (the
    trapezoid over [-10, 10] in Z with H_NODES nodes, exact to 1e-12 for v up to 100)
    and read by linear interpolation (error below 3e-7); v = 0 is the sigmoid itself."""

    _zg = np.arange(-H_ZMAX, H_ZMAX + H_STEP / 2, H_STEP)
    _zn = np.linspace(-10.0, 10.0, H_NODES)
    _zw = np.exp(-_zn ** 2 / 2)
    _zw = _zw / _zw.sum()

    def __init__(self, v):
        self.v = max(float(v), 0.0)
        if self.v > 0:
            sd = math.sqrt(self.v)
            tab = np.empty(len(self._zg))
            for a in range(0, len(self._zg), 2000):
                tab[a:a + 2000] = expit(self._zg[a:a + 2000, None] + sd * self._zn[None, :]) @ self._zw
            self.tab = tab

    def __call__(self, z):
        if self.v == 0:
            return expit(z)
        return np.interp(z, self._zg, self.tab)


def hexact(z, v):
    """E[sigmoid(z + sqrt(v) Z)] directly (for checking HTable)."""
    z = np.atleast_1d(np.asarray(z, float))
    if v <= 0:
        return expit(z)
    return expit(z[:, None] + math.sqrt(v) * HTable._zn[None, :]) @ HTable._zw


class Target:
    """What one target's eta is made of under hier's model (hier's own
    _Fit._target): coefficients on x (cols, coef), the prior mean m and variance v of
    the components no label touched, and the labeled items whose residuals it reads:
    its own item (kind 'own', factor 1) or the carriers of the folded group levels it
    shares (kind 'fold', factor k = su / s2e and residual variance su (1 - k) added to
    v), else none (kind 'new'). The floor c and slip of the prediction."""

    def __init__(self, fit, model, inp):
        subject, item = inp
        tg = fit._target(subject, item)
        if tg.s2t:
            raise NotImplementedError("an untouched Student-t level")
        self.cols = np.array(sorted(tg.a), np.int64)
        self.coef = np.array([tg.a[c] for c in self.cols], float)
        self.m, v = float(tg.m), float(tg.v)
        self.items, self.factors = [], []
        if tg.j is not None:
            self.kind = "own"
            self.items, self.factors = [int(tg.j)], [1.0]
        elif tg.fold:
            self.kind = "fold"
            for jj, su in sorted(tg.fold.items()):
                k = su / float(fit.prob.s2e[jj])
                self.items.append(int(jj))
                self.factors.append(k)
                v += su * (1 - k)
        else:
            self.kind = "new"
        if len(self.items) > 2:
            raise NotImplementedError("more than two folded carriers")
        self.v = v
        self.H = HTable(v)
        self.c = model.floor_item(item_key(item), item)
        self.slip = model.hyper.slip if model.cfg.slip else 0.0

    def u(self, X):
        return self.m + X[:, self.cols] @ self.coef if len(self.cols) else np.full(len(X), self.m)

    def F(self, X, Q, E):
        """E[sigmoid(eta_t) | x] per row of X; Q: {item: (rows, K) node weights},
        E: the items' nodes."""
        u = self.u(X)
        if self.kind == "new":
            return self.H(u)
        if len(self.items) == 1:
            j, k = self.items[0], self.factors[0]
            z = u[:, None] - k * E[j][None, :]
            return (Q[j] * self.H(z)).sum(1)
        (j1, j2), (k1, k2) = self.items, self.factors
        z = u[:, None, None] - k1 * E[j1][None, :, None] - k2 * E[j2][None, None, :]
        return np.einsum("nk,nl,nkl->n", Q[j1], Q[j2], self.H(z))

    def p(self, ef):
        """The prediction from E[sigmoid(eta_t)], as HierPredictor.combine and
        predict clip it."""
        return _probability(self.c + (1 - self.c - self.slip) * float(ef))


def systematic(w, rng):
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), pos), n - 1)


def ess_of(logw):
    w = np.exp(logw - np.max(logw))
    return float(w.sum() ** 2 / (w * w).sum())


def khat(logw):
    """Pareto k-hat of importance weights (Vehtari et al., PSIS; the Zhang and
    Stephens generalized Pareto fit to the largest min(0.2 n, 3 sqrt(n)) weights)."""
    lw = np.sort(np.asarray(logw, float))
    n = len(lw)
    M = int(min(0.2 * n, 3 * math.sqrt(n)))
    if M < 5:
        return None
    w = np.exp(lw - lw[-1])
    x = w[-M:] - w[-M - 1]
    x = np.sort(x[x > 0])
    n2 = len(x)
    if n2 < 5:
        return None
    m_est = 30 + int(n2 ** 0.5)
    b = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b /= 3 * x[int(n2 / 4 + 0.5) - 1]
    b += 1 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(1)
    L = n2 * (np.log(-(b / k)) - k - 1)
    wt = 1 / np.exp(L - L[:, None]).sum(1)
    ok = wt >= 10 * np.finfo(float).eps
    wt, b = wt[ok], b[ok]
    wt /= wt.sum()
    bp = float((b * wt).sum())
    kp = float(np.log1p(-bp * x).mean())
    return (n2 * kp + 10 * 0.5) / (n2 + 10)


def laplace(fit):
    """The Gaussian hier fits: mode, Cholesky factor of its covariance."""
    x0 = np.asarray(fit.post.x, float)
    C = np.asarray(fit.post.cov, float)
    C = 0.5 * (C + C.T)
    return x0, np.linalg.cholesky(C)


def smc(ex, targets, rng, n=PARTICLES, tau=TAU, moves=MOVES, final_moves=FINAL_MOVES):
    """One SMC replicate: {'F': mean E[sigmoid(eta_t) | x] per target over every
    particle of every move at the posterior, 'steps': [(beta, ESS)], 'acc': mean
    acceptance at the posterior, 'rho' at the end, 'evals' of the log posterior}.

    Start: n draws from the Laplace Gaussian q = N(x0, LL'). Tempering pi_b
    proportional to q^(1-b) pi^b: the next b keeps an ESS of tau n (or is 1 if 1
    does), then systematic resampling and `moves` Metropolis moves (final_moves at
    b = 1). The move is x' = x0 + rho (x - x0) + sqrt(1 - rho^2) L xi, reversible with
    respect to q, so it is accepted with probability min(1, (w(x') / w(x))^b), w = pi /
    q; rho adapts to keep the acceptance within [ACC_LO, ACC_HI]."""
    x0, L = laplace(ex.fit)
    keep = sorted({j for t in targets for j in t.items})
    kidx = {j: a for a, j in enumerate(keep)}
    Ek = {j: ex.E[j] for j in keep}

    def evaluate(Zw, with_q):
        X = x0 + Zw @ L.T
        lp, Q = ex.logpost(X, keep if (with_q and keep) else None)
        return X, lp - (-0.5 * (Zw * Zw).sum(1)), Q

    evals0 = ex.evals
    Zw = rng.standard_normal((n, ex.p))
    X, lw, _ = evaluate(Zw, False)
    if not np.all(np.isfinite(lw)):
        raise FloatingPointError("non-finite log weights")
    beta, logW, steps, rho = 0.0, np.zeros(n), [], RHO0
    acc_final, sums, count = [], np.zeros(len(targets)), 0
    while True:
        def ess_at(b):
            return ess_of(logW + (b - beta) * lw)
        if ess_at(1.0) >= tau * n:
            nb = 1.0
        else:
            lo, hi = beta, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ess_at(mid) >= tau * n else (lo, mid)
            nb = max(lo, beta + 1e-6)
        logW = logW + (nb - beta) * lw
        beta = nb
        steps.append((r(beta), r(ess_of(logW) / n, 4)))
        w = np.exp(logW - logW.max())
        idx = systematic(w / w.sum(), rng)
        Zw, X, lw = Zw[idx], X[idx], lw[idx]
        logW = np.zeros(n)
        final = beta >= 1.0
        Fcur = None
        if final:
            _, Q = ex.logpost(X, keep if keep else None)
            Fcur = np.column_stack([t.F(X, {j: Q[:, kidx[j]] for j in t.items}, Ek)
                                    for t in targets]) if targets else np.zeros((n, 0))
        for _ in range(final_moves if final else moves):
            xi = rng.standard_normal(Zw.shape)
            Zp = rho * Zw + math.sqrt(1 - rho * rho) * xi
            Xp, lwp, Qp = evaluate(Zp, final)
            a = np.log(rng.random(n)) < beta * (lwp - lw)
            Zw[a], X[a], lw[a] = Zp[a], Xp[a], lwp[a]
            rate = float(a.mean())
            if rate > ACC_HI:
                rho = max(0.0, 1 - (1 - rho) * 1.25)
            elif rate < ACC_LO:
                rho = min(0.995, 1 - (1 - rho) / 1.5)
            if final:
                acc_final.append(rate)
                if targets:
                    if a.any():
                        Xa = Xp[a]
                        Qa = Qp[a] if Qp is not None else None
                        Fcur[a] = np.column_stack([t.F(Xa, {j: Qa[:, kidx[j]] for j in t.items}, Ek)
                                                   for t in targets])
                    sums += Fcur.sum(0)
                count += n
        if final:
            break
    return {"F": sums / max(count, 1), "steps": steps, "acc": r(np.mean(acc_final), 4),
            "rho": r(rho, 4), "evals": ex.evals - evals0}


def _log_t(u, df):
    """Sum over coordinates of the log Student-t density (df degrees of freedom)."""
    c = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    return u.shape[1] * c - 0.5 * (df + 1) * np.log1p(u * u / df).sum(1)


def defensive_is(ex, targets, rng, n=IS_DRAWS, batches=IS_BATCHES, alpha=IS_ALPHA, df=IS_DF):
    """Importance sampling from q = (1 - alpha) N(x0, LL') + alpha (x0 + L t), t with
    independent Student-t(df) coordinates. -> {'F': E[sigmoid(eta_t)] per target,
    'se': its delta-method SE, 'batches': (batches, targets) estimates of the
    independent batches, 'ess', 'khat', 'max_weight_x_n', 'tail_share': the weight
    the t component's draws carry, 'psi_w': the normalised weights and 'Fd' the
    per-draw E[sigmoid(eta_t) | x] (for the delta-method SE of Brier differences)}."""
    x0, L = laplace(ex.fit)
    keep = sorted({j for t in targets for j in t.items})
    kidx = {j: a for a, j in enumerate(keep)}
    Ek = {j: ex.E[j] for j in keep}
    d = ex.p
    lws, Fs, comp, bat = [], [], [], []
    per = n // batches
    for k in range(batches):
        for a in range(0, per, 2000):
            m = min(2000, per - a)
            wide = rng.random(m) < alpha
            u = rng.standard_normal((m, d))
            if wide.any():
                g = rng.chisquare(df, (int(wide.sum()), d))
                u[wide] = u[wide] / np.sqrt(g / df)
            lq = np.logaddexp(math.log1p(-alpha) - 0.5 * (u * u).sum(1) - 0.5 * d * math.log(2 * math.pi),
                              math.log(alpha) + _log_t(u, df))
            X = x0 + u @ L.T
            lp, Q = ex.logpost(X, keep if keep else None)
            lws.append(lp - lq)
            Fs.append(np.column_stack([t.F(X, {j: Q[:, kidx[j]] for j in t.items}, Ek)
                                       for t in targets]) if targets else np.zeros((m, 0)))
            comp.append(wide)
            bat.append(np.full(m, k))
    lw, F = np.concatenate(lws), np.concatenate(Fs)
    comp, bat = np.concatenate(comp), np.concatenate(bat)
    if not np.all(np.isfinite(lw)):
        raise FloatingPointError("non-finite log weights")
    w = np.exp(lw - lw.max())
    w /= w.sum()
    est = w @ F
    se = np.sqrt(((w[:, None] * (F - est[None, :])) ** 2).sum(0))
    B = []
    for k in range(batches):
        wk = w[bat == k]
        B.append(wk @ F[bat == k] / wk.sum())
    return {"F": est, "se": se, "batches": np.array(B), "ess": r(1 / (w * w).sum() / len(w), 4),
            "khat": r(khat(lw), 3), "max_weight_x_n": r(w.max() * len(w), 3),
            "tail_share": r(w[comp].sum(), 5), "psi_w": w, "Fd": F}


def delta_se_diff(isr, targets, ys):
    """Delta-method Monte Carlo SE of mean_t (p_x,t - y_t)^2 - (p_exact,t - y_t)^2: to
    first order the difference moves by -2 (p_exact,t - y_t) (1 - c_t - slip) / T
    per unit of E[sigmoid(eta_t)], whatever predictor x is, so one SE serves all."""
    T = len(targets)
    pe = np.array([t.p(f) for t, f in zip(targets, isr["F"])])
    g = np.array([-2 * (p - y) * (1 - t.c - t.slip) / T for t, p, y in zip(targets, pe, ys)])
    psi = (isr["Fd"] - isr["F"][None, :]) @ g
    return float(math.sqrt(((isr["psi_w"] * psi) ** 2).sum()))


# --- targets and checkpoints ----------------------------------------------------------------------

def input_digest(inp):
    return hashlib.sha256(PD.O._canon(inp).encode()).hexdigest()[:16]


def pick_targets(slots, n=N_TARGETS):
    """[(slot index, input, label, response position)]: pairs in digest order, round
    robin, each pair's evaluation responses in digest order."""
    ranked = []
    for si, s in enumerate(slots):
        resp, seen = [], {}
        for inp, y, key in s.targets:
            k = seen.get(key, 0)
            seen[key] = k + 1
            resp.append((stable_hash(SPLIT_SEED, TARGET_SALT, s.subject_id, s.benchmark_id, key, k),
                         inp, y, len(resp)))
        resp.sort(key=lambda t: t[0])
        ranked.append((stable_hash(SPLIT_SEED, TARGET_SALT, "pair", s.subject_id, s.benchmark_id),
                       si, resp))
    ranked.sort(key=lambda t: t[0])
    out, depth = [], 0
    while len(out) < n and any(depth < len(r_[2]) for r_ in ranked):
        for _, si, resp in ranked:
            if depth < len(resp) and len(out) < n:
                _, inp, y, pos = resp[depth]
                out.append((si, inp, y, pos))
        depth += 1
    return out


def bundle(bench):
    prior, hyper = LC.bundle(bench, 0.0)
    return prior, replace(hyper, **LEVEL)


def own_fn(model):
    return PD.OwnOnly(model.predict)


def population(model, labeled, inputs, index, slots):
    """hier on every scored target of the checkpoint: per pair Brier (as
    pooling_decomposition stores it) and the pair-averaged Brier."""
    preds = np.array([model.predict(inp, labeled) for inp in inputs])[index]
    out, at = [], 0
    for s in slots:
        y = np.array([t[1] for t in s.targets], float)
        out.append(float(np.mean((preds[at:at + len(y)] - y) ** 2)))
        at += len(y)
    return out


def stored_ship(bench, scope):
    path = os.path.join(PD_ROWS, f"{bench}.{scope}.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        st = json.load(f)
    return st, sha256_file(path)


def score_checkpoint(bench, scope, budget, slots, cps, subjects, prior, hyper, log=print):
    """Every number of one checkpoint: the target rows, the exact predictive's
    replicates and the diagnostics."""
    t0 = time.time()
    b, labeled, inputs, index = next(c for c in cps if c[0] == budget)
    picks = pick_targets(slots)
    model = HierPredictor(prior, hyper)
    fit = model.fit_for(labeled)
    if not isinstance(fit, _Fit):
        raise RuntimeError("the fit was split by benchmark")
    p_hier = [model.predict(inp, labeled) for _, inp, _, _ in picks]
    lap = HierPredictor(prior, hyper, line=False)
    p_lap = [lap.predict(inp, labeled) for _, inp, _, _ in picks]
    own = own_fn(HierPredictor(prior, hyper))
    p_own = [own(inp, labeled) for _, inp, _, _ in picks]
    t_hier = time.time() - t0

    # population Brier, against the stored per-pair rows of pooling_decomposition
    pop = population(HierPredictor(prior, hyper), labeled, inputs, index, slots)
    st, st_sha = stored_ship(bench, scope)
    repro = None
    if st is not None:
        bi = list(BUDGETS).index(budget)
        stored = [row[bi] for row in st["res"]["ship"]["b"]]
        same_pairs = [m["subject"] for m in st["meta"]] == list(subjects)
        repro = {"rows_sha256": st_sha, "pairs_match": bool(same_pairs),
                 "max_abs": r(np.max(np.abs(np.array(pop) - np.array(stored))), 12),
                 "bit_identical": bool(same_pairs and pop == stored)}

    ex = Exact(fit)
    targets = [Target(fit, model, inp) for _, inp, _, _ in picks]
    ys = [int(y) for _, _, y, _ in picks]

    # checks of the log posterior: against hier's Problem.state, and a finer grid
    rng = np.random.default_rng([SEED, BENCHES.index(bench), SCOPES.index(scope), budget, 0])
    x0, L = laplace(fit)
    Xc = np.vstack([x0[None, :], x0 + rng.standard_normal((40, ex.p)) @ L.T])
    mine, _ = ex.logpost(Xc)
    theirs = np.array([fit.prob.state(x).lp for x in Xc])
    fine, _ = Exact(fit, refine=GRID_CHECK).logpost(Xc)
    lp_check = {"vs_hier_max_abs": r(np.max(np.abs(mine - theirs)), 9),
                "vs_hier_spread": r(np.ptp(mine - theirs), 9),
                "vs_finer_grid_max_abs": r(np.max(np.abs(mine - fine)), 12),
                "grid_nodes": ex.K, "draws": len(Xc)}
    zt = np.linspace(-30, 30, 2001)
    h_err = max([float(np.max(np.abs(t.H(zt) - hexact(zt, t.v)))) for t in targets[:5]])

    # the estimate: defensive importance sampling
    rng = np.random.default_rng([SEED, BENCHES.index(bench), SCOPES.index(scope), budget, 2])
    t1 = time.time()
    isr = defensive_is(ex, targets, rng, n=IS_DRAWS, batches=IS_BATCHES)
    mc_delta = delta_se_diff(isr, targets, ys)
    is_secs = round(time.time() - t1, 1)
    log(f"    importance sampling: ESS {isr['ess']}, k-hat {isr['khat']}, "
        f"t share {isr['tail_share']}, {is_secs} s")
    # the second estimate: SMC
    reps = []
    for k in range(REPLICATES):  # noqa: B007
        rng = np.random.default_rng([SEED, BENCHES.index(bench), SCOPES.index(scope), budget, 1, k])
        t1 = time.time()
        out = smc(ex, targets, rng, n=PARTICLES, tau=TAU, moves=MOVES, final_moves=FINAL_MOVES)
        out["secs"] = round(time.time() - t1, 1)
        reps.append(out)
        log(f"    SMC replicate {k}: steps {out['steps']}, acc {out['acc']}, {out['secs']} s")
    Fr = np.array([o["F"] for o in reps])            # (replicates, targets)

    rows = []
    for a, ((si, inp, y, pos), t) in enumerate(zip(picks, targets)):
        ps = [t.p(f) for f in Fr[:, a]]
        rows.append({"pair": si, "subject": subjects[si], "input": input_digest(inp),
                     "pos": pos, "y": int(y), "kind": t.kind, "n_items_read": len(t.items),
                     "floor": r(t.c, 6),
                     "hier": p_hier[a], "laplace": p_lap[a], "own": p_own[a],
                     "exact": t.p(isr["F"][a]),
                     "exact_se": r((1 - t.c - t.slip) * isr["se"][a], 9),
                     "exact_batches": [r(t.p(f), 9) for f in isr["batches"][:, a]],
                     "exact_smc": t.p(np.mean(Fr[:, a])),
                     "exact_smc_se": r(np.std(ps, ddof=1) / math.sqrt(len(ps)), 9),
                     "exact_smc_reps": [r(x, 9) for x in ps]})
    return {"bench": bench, "scope": scope, "budget": budget,
            "labels": len(fit.prob.y), "columns": ex.p, "items_labeled": ex.n_items,
            "floored_labels": int((fit.prob.c > 0).sum()), "pairs": len(slots),
            "rows": rows,
            "population": {"per_pair_brier": [r(x, 12) for x in pop],
                           "brier": r(np.mean(pop), 9), "repro": repro},
            "lp_check": lp_check, "htable_max_err": r(h_err, 12),
            "is": {"draws": IS_DRAWS, "batches": IS_BATCHES, "ess": isr["ess"], "khat": isr["khat"],
                   "max_weight_x_n": isr["max_weight_x_n"], "tail_share": isr["tail_share"],
                   "mc_se_diff_delta": r(mc_delta, 8), "secs": is_secs},
            "smc": [{k: v for k, v in o.items() if k != "F"} for o in reps],
            "hier_converged": bool(fit.converged), "hier_secs": round(t_hier, 2),
            "secs": round(time.time() - t0, 1), "utc": utcnow()}


def checkpoints_of(bench, scope):
    """(slots, checkpoints, public subject id of each slot) of the dense run, as
    pooling_decomposition plays it."""
    run = LC.dense(bench)
    slots, cps, _ = PD.checkpoints(run, scope, PD.EVAL_CAP.get(bench), SPLIT_SEED)
    return slots, cps, [PD.O._entry(e)[0].subject_id for e in run]


# --- statistics ---------------------------------------------------------------------------------------

def brier_diff(rows, x, ref="exact"):
    y = np.array([w["y"] for w in rows], float)
    return (np.array([w[x] for w in rows]) - y) ** 2 - (np.array([w[ref] for w in rows]) - y) ** 2


def boot(d, clusters, rng, n=BOOTS):
    """(target bootstrap (lo, hi, se), pair-cluster bootstrap (lo, hi, se)): percentile
    intervals of the mean of d."""
    d = np.asarray(d, float)
    m = len(d)
    ti = rng.integers(0, m, (n, m))
    tb = d[ti].mean(1)
    cl = np.asarray(clusters)
    uc = np.unique(cl)
    sums = np.array([d[cl == c].sum() for c in uc])
    cnts = np.array([(cl == c).sum() for c in uc], float)
    ci = rng.integers(0, len(uc), (n, len(uc)))
    cb = sums[ci].sum(1) / cnts[ci].sum(1)
    q = lambda v: (r(np.percentile(v, 2.5), 6), r(np.percentile(v, 97.5), 6), r(np.std(v, ddof=1), 6))
    return q(tb), q(cb)


def mc_se_batches(rows, x):
    """Monte Carlo SE of mean (p_x - y)^2 - (p_exact - y)^2 from the spread of the
    difference over the independent importance-sampling batches."""
    y = np.array([w["y"] for w in rows], float)
    px = np.array([w[x] for w in rows])
    E = np.array([w["exact_batches"] for w in rows])          # (targets, batches)
    Dr = (((px - y) ** 2)[:, None] - (E - y[:, None]) ** 2).mean(0)
    return r(np.std(Dr, ddof=1) / math.sqrt(len(Dr)), 8)


def summarise_checkpoint(ck, rng):
    rows = ck["rows"]
    y = np.array([w["y"] for w in rows], float)
    out = {"targets": len(rows), "pairs_sampled": len({w["pair"] for w in rows}),
           "kinds": {k: sum(w["kind"] == k for w in rows) for k in ("own", "fold", "new")},
           "brier": {x: r(np.mean((np.array([w[x] for w in rows]) - y) ** 2))
                     for x in ("hier", "laplace", "own", "exact", "exact_smc")},
           "population_brier_hier": ck["population"]["brier"],
           "base_rate": r(y.mean(), 4)}
    for x in ("hier", "laplace", "own"):
        d = brier_diff(rows, x)
        tb, cb = boot(d, [w["pair"] for w in rows], rng)
        mcb = mc_se_batches(rows, x)
        mcd = ck["is"]["mc_se_diff_delta"]
        out[f"{x} - exact"] = {"mean": r(d.mean()), "target_boot": tb, "cluster_boot": cb,
                               "mc_se": r(max(mcb, mcd), 8), "mc_se_delta": mcd, "mc_se_batches": mcb}
    d = brier_diff(rows, "hier", ref="own")
    tb, cb = boot(d, [w["pair"] for w in rows], rng)
    out["hier - own"] = {"mean": r(d.mean()), "target_boot": tb, "cluster_boot": cb}
    ph = np.array([w["hier"] for w in rows])
    pe = np.array([w["exact"] for w in rows])
    pl = np.array([w["laplace"] for w in rows])
    toward = np.sign(pe - 0.5) * (pe - ph)          # > 0: hier nearer 0.5 than exact
    out["abs_diff"] = {"hier": r(np.mean(np.abs(ph - pe))), "laplace": r(np.mean(np.abs(pl - pe))),
                       "hier_max": r(np.max(np.abs(ph - pe))), "laplace_max": r(np.max(np.abs(pl - pe)))}
    out["hier_toward_half"] = r(np.mean(toward))
    se = np.array([w["exact_se"] for w in rows])
    out["exact_se"] = {"median": r(np.median(se), 7), "max": r(np.max(se), 7)}
    sse = np.array([w["exact_smc_se"] for w in rows])
    ps = np.array([w["exact_smc"] for w in rows])
    zz = (ps - pe) / np.sqrt(sse ** 2 + se ** 2)
    out["smc_vs_is"] = {"max_abs": r(np.max(np.abs(ps - pe))), "mean_abs": r(np.mean(np.abs(ps - pe))),
                        "mean_diff": r(np.mean(ps - pe)),
                        "max_abs_z": r(np.max(np.abs(zz)), 3),
                        "share_abs_z_above_3": r(np.mean(np.abs(zz) > 3), 3),
                        "smc_se_median": r(np.median(sse), 7)}
    out["is"] = {k: ck["is"][k] for k in ("ess", "khat", "max_weight_x_n", "tail_share")}
    return out


def gate(summary):
    """G1 as the plan fixes it, read on GATE_BENCH, GATE_SCOPE, GATE_BUDGET: the
    difference hier - exact and its pair-cluster bootstrap interval."""
    k = ck_key(GATE_BENCH, GATE_SCOPE, GATE_BUDGET)
    if k not in summary:
        return {"rule": GATE_TEXT, "read_on": k, "verdict": None, "why": "not scored"}
    d = summary[k]["hier - exact"]
    lo, hi, _ = d["cluster_boot"]
    excl = lo > 0 or hi < 0
    opens = d["mean"] >= GATE_MIN and excl
    out = {"rule": GATE_TEXT, "read_on": k, "difference": d["mean"], "interval_95_cluster": [lo, hi],
           "interval_95_target": list(d["target_boot"][:2]), "mc_se": d["mc_se"],
           "at_least_0.003": bool(d["mean"] >= GATE_MIN), "interval_excludes_0": bool(excl),
           "verdict": "stage 2 opens" if opens else "S1 closes"}
    alt = ck_key(GATE_BENCH, "benchmark", GATE_BUDGET)
    if alt in summary:
        a = summary[alt]["hier - exact"]
        alo, ahi, _ = a["cluster_boot"]
        out["same_rule_scope_benchmark"] = {
            "difference": a["mean"], "interval_95_cluster": [alo, ahi],
            "would_open": bool(a["mean"] >= GATE_MIN and (alo > 0 or ahi < 0))}
    return out


# --- stages -------------------------------------------------------------------------------------------

def load_out():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {}


def provenance(extra=None):
    script = os.path.abspath(__file__)
    inputs = {}
    for b in PUBLIC:
        for t in DATA_TABLES:
            p = os.path.join(D.DATA_DIR, b, t)
            if os.path.exists(p):
                inputs[os.path.relpath(p, ROOT) if p.startswith(ROOT) else p] = sha256_file(p)
    for b in BENCHES:
        for sc in SCOPES:
            p = os.path.join(PD_ROWS, f"{b}.{sc}.json")
            if os.path.exists(p):
                inputs[os.path.relpath(p, ROOT)] = sha256_file(p)
    out = {"script": os.path.relpath(script, ROOT), "script_sha256": sha256_file(script),
           "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
           "lib_sha256": {f: sha256_file(os.path.join(ROOT, f)) for f in LIB_FILES},
           "inputs_sha256": inputs, "python": platform.python_version(), "numpy": np.__version__,
           "machine": platform.machine(), "utc": utcnow()}
    out.update(extra or {})
    return out


def design():
    return {"benches": BENCHES, "gate": [GATE_BENCH, GATE_SCOPE, GATE_BUDGET, GATE_MIN],
            "scopes": SCOPES, "budgets": CHECK_BUDGETS, "targets": N_TARGETS, "split_seed": SPLIT_SEED,
            "eval_cap": {b: PD.EVAL_CAP.get(b) for b in BENCHES}, "level": LEVEL,
            "prior": "level_calibration.bundle(benchmark, 0.0) with LEVEL: leave-one-benchmark-out",
            "smc": {"particles": PARTICLES, "replicates": REPLICATES, "tau": TAU, "moves": MOVES,
                    "final_moves": FINAL_MOVES, "rho0": RHO0, "acc": [ACC_LO, ACC_HI]},
            "is": {"draws": IS_DRAWS, "batches": IS_BATCHES, "alpha": IS_ALPHA, "df": IS_DF},
            "grid": {"U": GRID_U, "h_max": H_MAX, "h_frac": H_FRAC, "check_refine": GRID_CHECK},
            "htable": {"zmax": H_ZMAX, "step": H_STEP, "nodes": H_NODES}, "boots": BOOTS}


def check_level():
    got = PD.model_level()
    if got != LEVEL:
        raise SystemExit(f"submission/model.py LEVEL is {got}, this check's {LEVEL}")


def stage_score(only=None):
    check_level()
    res = load_out()
    res.setdefault("what", "S1 stage 1 (docs/plans/p2_final_plan.md section 3): hier against the "
                           "exact posterior predictive of its own model at dense B1 and B3")
    res["design"] = design()
    res.setdefault("checkpoints", {})
    res.setdefault("commands", [])
    res["commands"].append(" ".join(["python", "experiments/dense_b1_check.py"] + sys.argv[1:]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for bench in BENCHES:
            for scope in SCOPES:
                if only and f"{bench}.{scope}" not in only and bench not in only:
                    continue
                todo = [B for B in CHECK_BUDGETS if ck_key(bench, scope, B) not in res["checkpoints"]]
                if not todo:
                    continue
                prior, hyper = bundle(bench)
                slots, cps, subjects = checkpoints_of(bench, scope)
                for B in todo:
                    print(f"{ck_key(bench, scope, B)}: scoring", flush=True)
                    ck = score_checkpoint(bench, scope, B, slots, cps, subjects, prior, hyper,
                                          log=lambda s: print(s, flush=True))
                    res["checkpoints"][ck_key(bench, scope, B)] = ck
                    res["complete"] = all(ck_key(b, s, x) in res["checkpoints"]
                                          for b in BENCHES for s in SCOPES for x in CHECK_BUDGETS)
                    res["provenance"] = provenance()
                    write_json(OUT, res)
                    print(f"  done in {ck['secs']} s", flush=True)


def stage_summarise():
    res = load_out()
    if not res.get("checkpoints"):
        raise SystemExit(f"nothing scored in {OUT}")
    rng = np.random.default_rng(BOOT_SEED)
    summary = {}
    for b in BENCHES:
        for s in SCOPES:
            for B in CHECK_BUDGETS:
                k = ck_key(b, s, B)
                if k in res["checkpoints"]:
                    summary[k] = summarise_checkpoint(res["checkpoints"][k], rng)
    res["summary"] = summary
    res["gate"] = gate(summary)
    cks = res["checkpoints"].values()
    res["checks"] = {
        "level_equals_model": PD.model_level() == LEVEL,
        "population_repro_bit_identical": {k: (c["population"]["repro"] or {}).get("bit_identical")
                                           for k, c in res["checkpoints"].items()},
        "lp_vs_hier_max_abs": r(max(c["lp_check"]["vs_hier_max_abs"] for c in cks), 9),
        "lp_vs_finer_grid_max_abs": r(max(c["lp_check"]["vs_finer_grid_max_abs"] for c in cks), 12),
        "htable_max_err": r(max(c["htable_max_err"] for c in cks), 12),
        "hier_all_converged": all(c["hier_converged"] for c in cks),
        "smc_vs_is_max_abs_z": {k: v["smc_vs_is"]["max_abs_z"] for k, v in summary.items()},
        "is_khat": {k: v["is"]["khat"] for k, v in summary.items()},
    }
    res["complete"] = all(ck_key(b, s, x) in res["checkpoints"]
                          for b in BENCHES for s in SCOPES for x in CHECK_BUDGETS)
    res.setdefault("commands", []).append("python experiments/dense_b1_check.py summarise")
    res["provenance"] = provenance({"summarised_utc": utcnow()})
    write_json(OUT, res)
    g = res["gate"]
    print(json.dumps(g, indent=1))
    for k, v in summary.items():
        d = v["hier - exact"]
        print(f"{k}: hier {v['brier']['hier']:.4f} exact {v['brier']['exact']:.4f} "
              f"diff {d['mean']:+.5f} cluster {d['cluster_boot'][:2]} target {d['target_boot'][:2]} "
              f"mc {d['mc_se']} |dp| {v['abs_diff']['hier']:.4f} toward 0.5 {v['hier_toward_half']:+.4f}")


def stage_status():
    res = load_out()
    have = sorted(res.get("checkpoints", {}))
    print(f"{len(have)} of {len(BENCHES) * len(SCOPES) * len(CHECK_BUDGETS)} checkpoints: {have}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("score", "summarise", "status"))
    ap.add_argument("--only", nargs="*", help="benchmark or benchmark.scope to score")
    ap.add_argument("--out", help="write here instead of results/dense_b1_check.json (trials)")
    ap.add_argument("--quick", action="store_true",
                    help="a trial: 20,000 IS draws in 4 batches, SMC 2 x 500 particles (with --out only)")
    a = ap.parse_args(argv)
    global OUT, PARTICLES, REPLICATES, IS_DRAWS, IS_BATCHES, FINAL_MOVES
    if a.quick and not a.out:
        raise SystemExit("--quick needs --out: the trial must not write results/")
    if a.out:
        OUT = os.path.abspath(a.out)
    if a.quick:
        PARTICLES, REPLICATES, IS_DRAWS, IS_BATCHES, FINAL_MOVES = 500, 2, 20000, 4, 10
    if a.stage == "score":
        stage_score(a.only)
    elif a.stage == "summarise":
        stage_summarise()
    else:
        stage_status()


if __name__ == "__main__":
    main()
