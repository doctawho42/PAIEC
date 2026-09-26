"""Calibrating for the hidden test: which predictor and level prior to ship.

The first formative feedback (the shipped Predictor, commit b68492c; 9 pairs,
ALC 0.2113, testlike.FEEDBACK) says the hidden pairs sit far below the public
ones (B31 reads as a pair-logit mean near -1.6) and that the shipped attribute
prior placed strong 2025-26 subjects near p = 0.75 on them (B0 0.359). This
script chooses what to ship for that distribution, on paired runs:

  primary    test-like runs (paiec.testlike, Regime() at its defaults, seed 2:
             the runs of experiments/testlike_check.py's check phase), runs
             0..N/2-1 to select and N/2..N-1 to confirm (--runs N)
  guard      public formative-like runs (official.sample_run, seed 0, split
             scope 'pair': the runs of experiments/hier_eval.py), benchmark-first
             and pair-uniform; a candidate may lose at most GUARD ALC against
             the shipped Predictor on either
  sensitive  test-like runs under other regime knobs (seed 3, SENS): the level
             centre, which one run of feedback does not identify (level_mean
             -1.2 and -2.0), item structure (groups merged at random, no
             strata), no date shift (the subjects' own dates)

Everything a predictor learns offline is fitted leave-one-parent-benchmark-out
for every target: one model per parent in the run, its attribute prior and
empirical-Bayes hyperparameters fitted without that parent (paiec.prior.build,
predict.fit_prior), dispatched on the anonymous benchmark_id, so predict never
sees a name. On public runs the parent is the benchmark itself.

Candidates
  smoothed          Beta(2,2) on the pair's own labels
  Predictor         the shipped Predictor (paiec.predict, v_a 2)
  P off= sc= va=    the Predictor with a level fix: prior mean scale * m + off
                    (m its attribute mean, accuracy-logit scale) and prior
                    variance v_a. v_a is a constructor argument; the mean is
                    changed by overriding prior_mean (as testlike_check's
                    Variant does), so the labels update from the moved prior
                    (not a wrap of the output). All variants share one Evidence
                    and IRT fit per checkpoint (PredictorGrid); the variant
                    off=0, sc=1, va=2 is checked to be the shipped Predictor
                    call for call (--verify)
  hier              paiec.hier at its target-LOBO defaults
  hier G ...        hier with the level prior set for the test: mu0 and
                    sigma_mu (item-level scale, replacing the fitted ones) and
                    attr_scale (the attribute standing's mean times this); G is
                    a Gaussian level, T a Student-t level with 3 df (sigma_mu
                    then its scale), the rest of the hyperparameters as fitted
  hierEB-<m>@<base> pair-level empirical Bayes on top of a base hier config: at
                    each checkpoint the level prior's centre (m 'c'), scale
                    ('s') or both ('cs') re-estimated from every pair in
                    `labeled`, all benchmarks (EBHier); B0 is the base's

Statistics. Mean ALC over runs (a run's ALC is the mean over its pairs), paired
differences against the shipped Predictor with three SEs: over runs, a cluster
bootstrap (2,000 resamples, a ratio estimator: a cluster brings its weight with
it) with (parent, subject) as the cluster on test-like runs (testlike.
cluster_key: pseudo-benchmarks of one parent overlap) and (benchmark, subject)
on public ones, and the same bootstrap stratified by parent benchmark; plus the
range leaving one parent out. Runs are scored by a harness that replays
paiec.official's own _slots and _acquire and evaluates every candidate on the
same checkpoints (fresh instances per checkpoint, one worker, deduplicated
inputs, no argument copies); --verify checks it against official.run_official
pair for pair.

Stages (tasks are pure functions of their key, checkpointed to --out as they
finish; --resume skips finished ones). Each stage's candidates follow from the
stored results of the earlier ones, so they run in this order:
  grid     selection half of the test-like runs: base (smoothed, the Predictor
           grid, hier default) and the first Gaussian hier grid (GRID_*)
  grid2    its extension (GRID2_*, P_EXT), added when the first grid's best sat
           on its edges
  r1base   the references and every Predictor variant on the public runs
  eb, t3   empirical-Bayes and Student-t variants on the selection half
  prob     the Predictor variants with robust_fit_ab (P_ROB), added when
           fit_ab was found to diverge
  eb       again, for the EB configs on moderately shifted bases
  screen   hier configs against the guard on 40 public runs (screen_configs)
  confirm  the shortlist on the confirmation half, the full public runs, the
           sensitivity regimes and dense public runs
  final    the rule's choice and the best guarded hier configs on the
           sensitivity regimes and dense runs, where confirm had not scored them
  --summarise packs the rows (compact_raw) and rebuilds the summary: the
  selection with its guard, the shipped config (SHIP) paired with the
  alternatives, the posterior-predictive check against the feedback, the
  sensitivities, EB's estimates by regime, fit_ab's divergences and latency.

Run: python experiments/level_calibration.py --verify
     for s in grid grid2 r1base eb t3 prob eb screen confirm final; do
       python experiments/level_calibration.py --stage $s --resume --jobs 6; done
     python experiments/level_calibration.py --summarise
(3 h 11 min on six processes of a shared machine; docs/findings.md, "Calibrating
for the hidden test", reads the summary)
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paiec import baselines as B  # noqa: E402
from paiec import data as D  # noqa: E402
from paiec import official as O  # noqa: E402
from paiec import prior as PR  # noqa: E402
from paiec import testlike as T  # noqa: E402
from paiec.evaluator import BUDGETS, WEIGHTS  # noqa: E402
from paiec.fitting import fit_ab, sig  # noqa: E402
from paiec.hier import HierPredictor  # noqa: E402
from paiec.mcq import floor_of  # noqa: E402
from paiec.predict import (HI, LO, Predictor, _text, fit_prior, item_key,  # noqa: E402
                           records, subject_key)

OUT = os.path.join(ROOT, "results", "level_calibration.json")
TESTLIKE_CHECK = os.path.join(ROOT, "results", "testlike_check.json")
BOOTS = 2000
GUARD = 0.003
PRED, HIER, SMOOTHED = "Predictor", "hier", "smoothed"

#: regime key -> (kind, seed, Regime overrides or R1 weighting)
REGIMES = {
    "tl": ("testlike", 2, {}),
    "r1b": ("r1", 0, "benchmark"),
    "r1p": ("r1", 0, "pair"),
    "tl lm-1.2": ("testlike", 3, {"level_mean": -1.2}),
    "tl lm-2.0": ("testlike", 3, {"level_mean": -2.0}),
    "tl mix/whole": ("testlike", 3, {"kinds": ("mix", "whole")}),
    "tl no shift": ("testlike", 3, {"date_shift": 0.0}),
}
SENS = ["tl lm-1.2", "tl lm-2.0", "tl mix/whole", "tl no shift"]
DENSE = ("real_webagents", "researchcodebench")

#: the Predictor's level fix: prior mean scale * m + off, prior variance va
P_OFF = (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0)
P_SC = (0.5, 1.0)
P_VA = (2.0, 3.0, 4.5)
#: the Gaussian hier grid (item-level scale)
GRID_MU0 = (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0)
GRID_SM = (0.9, 1.3, 1.8, 2.5)
GRID_AS = (0.5, 0.75, 1.0)
#: empirical-Bayes prior widths (item-level scale): the run's centre about the
#: base mu0, and the log of the pairs' spread about the base's implied one
TAU_MU, TAU_S, EB_FLOOR = 1.0, 0.3, 0.5
SHORT_G = 9                         # guard-passing hier configs carried to confirmation

_pairs = _cat = None
_samplers, _bundles, _ppriors = {}, {}, {}


# --- names ---------------------------------------------------------------------------

def pname(off, sc, va, robust=False):
    if robust:
        return PRED + " rf" if (off, sc, va) == (0.0, 1.0, 2.0) else \
            f"Pr off={off:+.1f} sc={sc:.2f} va={va:.1f}"
    return PRED if (off, sc, va) == (0.0, 1.0, 2.0) else f"P off={off:+.1f} sc={sc:.2f} va={va:.1f}"


P_VARIANTS = [(o, s, v) for o in P_OFF for s in P_SC for v in P_VA]
P_NAMES = [pname(*v) for v in P_VARIANTS]
#: its extension, added when the first grid's best sat on its edges (a lower
#: offset, a weaker or no attribute standing)
P_EXT = [(o, s, v) for o in (-3.5, -3.0, -2.5, -2.0, -1.5) for s in (0.0, 0.25) for v in (2.0, 3.0)] \
    + [(o, s, v) for o in (-3.5, -3.0) for s in (0.5, 1.0) for v in (2.0, 3.0)]
P_EXT_NAMES = [pname(*v) for v in P_EXT]
#: the same with robust_fit_ab in place of fitting.fit_ab (names 'Pr ...'; the
#: shipped setting refitted robustly is 'Predictor rf'), added when fit_ab was
#: found to diverge under a moved prior
P_ROB = [(0.0, 1.0, 2.0)] + [(o, s, v) for o in (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5)
                             for s in (0.5, 1.0) for v in (2.0, 3.0)]
P_ROB_NAMES = [pname(*v, robust=True) for v in P_ROB]
P_ALL = P_NAMES + P_EXT_NAMES + P_ROB_NAMES


def gname(mu0, sm, a, nu=0):
    return f"hier {'T' if nu else 'G'} mu0={mu0:+.2f} sm={sm:.2f} as={a:.2f}"


def ebname(mode, base):
    return f"hierEB-{mode}@{base}"


GRID1_NAMES = [gname(m, s, a) for a in GRID_AS for s in GRID_SM for m in GRID_MU0]
#: the extension: hier's level is on the item-level scale, where the first
#: grid's lowest mu0 (-2.5) is about -1.15 on the accuracy-logit scale, and
#: the first grid's best sat on its edges (mu0 -2.5, attr_scale 0.5)
GRID2_MU0 = (-5.0, -4.0, -3.5, -3.0)
GRID2_SM = (1.3, 1.8, 2.5)
GRID2_AS = (0.25, 0.5, 1.0)
GRID2_NAMES = [gname(m, s, a) for a in GRID2_AS for s in GRID2_SM for m in GRID2_MU0]
GRID_NAMES = GRID1_NAMES + GRID2_NAMES


def parse(name):
    """A hier config from its name: (nu, overrides or None for the fitted
    defaults, EB mode or None)."""
    if name == HIER:
        return 0.0, None, None
    if name.startswith("hierEB-"):
        mode, base = name[len("hierEB-"):].split("@", 1)
        nu, ov, _ = parse(base)
        return nu, ov, mode
    kind, *kv = name[len("hier "):].split()
    d = dict(x.split("=") for x in kv)
    return (3.0 if kind == "T" else 0.0,
            {"mu0": float(d["mu0"]), "sigma_mu": float(d["sm"]), "attr_scale": float(d["as"])},
            None)


def family(name):
    if name == SMOOTHED:
        return "smoothed"
    if name == PRED:
        return "Predictor"
    if name.startswith("P "):
        return "Predictor fix"
    if name == HIER:
        return "hier default"
    if name.startswith("hierEB"):
        return "hier EB"
    return "hier T" if name.startswith("hier T") else "hier G"


# --- per-worker state ----------------------------------------------------------------

def pairs():
    global _pairs
    if _pairs is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _pairs = O.eligible(D.load_pairs())
    return _pairs


def catalogue():
    global _cat
    if _cat is None:
        _cat = T.build_catalogue(pairs())
    return _cat


def sampler(regime):
    if regime not in _samplers:
        _samplers[regime] = T.Sampler(catalogue(), T.Regime().with_(**REGIMES[regime][2]))
    return _samplers[regime]


def bundle(parent, nu):
    """(SubjectPrior, Hyper) fitted on every public pair outside `parent`."""
    key = (parent, float(nu))
    if key not in _bundles:
        _bundles[key] = PR.build(pairs(), (parent,), nu_mu=float(nu))
    return _bundles[key]


def ppred(parent):
    """The Predictor's attribute prior (coef, spec) without `parent`."""
    if parent not in _ppriors:
        train = [p for p in pairs() if p.benchmark_id != parent]
        _ppriors[parent] = fit_prior(train) if train else (None, None)
    return _ppriors[parent]


def init(bundles, ppriors):
    _bundles.update(bundles)
    _ppriors.update(ppriors)
    pairs()


def draw(regime, i):
    kind, seed, arg = REGIMES[regime]
    rng = np.random.default_rng([seed, i])
    if kind == "r1":
        return O.sample_run(pairs(), rng, weighting=arg)
    return sampler(regime).run(rng)


def dense(bench):
    return O.dense_run(pairs(), bench)


# --- predictors ----------------------------------------------------------------------

def robust_fit_ab(Z, Y, m_a=0.0, v_a=2.0, m_b=1.0, v_b=0.25, iters=100):
    """fitting.fit_ab's answer (the MAP of (a, b) and the Laplace variances at
    it, from the same clipped Hessian), found by Newton with step halving on
    the log posterior, which is concave, so every step is an ascent.

    fit_ab takes full Newton steps from the prior mean. When that mean sits far
    from the labels the first step overshoots into the flat tail of the
    logistic, where its clipped weights vanish, and it stops there: on one
    test-like pair at 27 of 31 with a prior mean of -1.88 (va 3) it returns
    a = -13.9 with the prior's variance, a prediction of 0.000 on a pair at
    0.83 (experiments/level_calibration.py, 'Pr' against 'P' rows)."""
    a, b = float(m_a), float(m_b)
    if len(Z) == 0:
        return a, b, v_a, v_b

    def logpost(a, b):
        eta = a + b * Z
        return float(-np.sum(Y * np.logaddexp(0, -eta) + (1 - Y) * np.logaddexp(0, eta))) \
            - (a - m_a) ** 2 / (2 * v_a) - (b - m_b) ** 2 / (2 * v_b)

    cur = logpost(a, b)
    for _ in range(iters):
        p = sig(a + b * Z)
        w = p * (1 - p)
        g = np.array([np.sum(Y - p) - (a - m_a) / v_a, np.sum((Y - p) * Z) - (b - m_b) / v_b])
        H = np.array([[np.sum(w) + 1 / v_a, np.sum(w * Z)],
                      [np.sum(w * Z), np.sum(w * Z * Z) + 1 / v_b]])
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        t = 1.0
        while True:
            na, nb = a + t * step[0], b + t * step[1]
            new = logpost(na, nb)
            if new >= cur - 1e-12 or t < 1e-10:
                break
            t /= 2
        moved = max(abs(na - a), abs(nb - b))
        a, b, cur = na, nb, max(new, cur)
        if moved < 1e-10:
            break
    p = np.clip(sig(a + b * Z), 1e-6, 1 - 1e-6)
    w = p * (1 - p)
    H = np.array([[np.sum(w) + 1 / v_a, np.sum(w * Z)],
                  [np.sum(w * Z), np.sum(w * Z * Z) + 1 / v_b]])
    C = np.linalg.inv(H + 1e-9 * np.eye(2))
    return float(a), float(b), float(C[0, 0]), float(C[1, 1])


class PredictorGrid(Predictor):
    """The shipped Predictor under every (off, sc, va) of `variants` at once.

    Each variant is Predictor._predict with prior mean sc * prior_mean + off and
    prior variance va; the Evidence (the per-benchmark IRT fit) does not depend
    on the prior, so it is built once per labeled list and shared. With
    `robust` the pair's (a, b) come from robust_fit_ab instead of fit_ab."""

    def __init__(self, coef, spec, variants, robust=False):
        super().__init__(coef, spec)
        self.variants = list(variants)
        self.fit = robust_fit_ab if robust else fit_ab

    def predict(self, input, labeled=None):
        try:
            return self._all(input, labeled)
        except Exception:
            self.failures += 1
            return np.full(len(self.variants), 0.5)

    def _all(self, input, labeled):
        subject, item = input
        text = _text(item.get("item_content"))
        bid = _text(item.get("benchmark_id"))
        ev = self._evidence_for(labeled)
        fit = ev.fit(bid)
        z = fit.z_of(item_key(item), text)
        sid = subject_key(subject)
        own = ev.ab.get(("own", bid, sid))
        if own is None:
            rows = [(fit.z[fit.index[r[2]]], r[3]) for r in ev.by_benchmark.get(bid, ())
                    if r[1] == sid]
            own = ev.ab[("own", bid, sid)] = (np.array([a for a, _ in rows]),
                                              np.array([y for _, y in rows], float))
        Z, Y = own
        m0 = self._prior_of(sid, subject)
        c = floor_of(text)
        out = np.empty(len(self.variants))
        for j, (off, sc, va) in enumerate(self.variants):
            key = (bid, sid, j)
            if key not in ev.ab:
                m = sc * m0 + off
                ab = self.fit(Z, Y, m, va, 1.0, self.v_b)
                ev.ab[key] = ab if all(map(math.isfinite, ab)) else (m, 1.0, va, self.v_b)
            a, b, v1, v2 = ev.ab[key]
            k = 1.0 / math.sqrt(1.0 + (math.pi / 8.0) * (v1 + v2 * z * z))
            p = c + (1 - c - self.slip) * sig(k * (a + b * z))
            out[j] = min(max(float(p), LO), HI) if math.isfinite(p) else 0.5
        return out


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(24)
_GH_W = _GH_W / _GH_W.sum()


class EBHier:
    """hier whose level prior is re-estimated from `labeled` at every checkpoint.

    Pairs in labeled (subject dict x benchmark_id, all benchmarks) are taken as
    eta_p = mu + m_p + u_p, u_p ~ N(0, S^2), with m_p the subject's prior mean
    standing (HierPredictor.theta_prior: shift + attr_scale x attributes) and
    each label Bernoulli(sigmoid(c eta_p)), c = (1 + pi T / 8)^-1/2 for the
    item variance T = sigma_d^2 + sigma_g^2 (items and floors integrated
    approximately). The MAP of (mu, log S) under mu ~ N(mu0, TAU_MU^2) and
    log S ~ N(log S0, TAU_S^2), S0^2 = sigma_mu^2 + sigma_theta^2 +
    sigma_delta^2 the base's implied pair spread, gives mu0 := mu ('c') and
    sigma_mu := sqrt(S^2 - sigma_theta^2 - sigma_delta^2), floored at EB_FLOOR
    ('s'). With no labels it is the base. `trace` logs (labels, mu0, sigma_mu)."""

    def __init__(self, prior, hyper, mode):
        what, *opts = mode.split(",")
        opts = dict(o.split("=") for o in opts)
        self.prior, self.base, self.mode = prior, hyper, what
        self.tau_mu = float(opts.get("tm", TAU_MU))
        self.tau_s = float(opts.get("ts", TAU_S))
        self.ref = HierPredictor(prior, hyper)
        self.models = {}
        self.trace = []
        self.failures = self.unconverged = 0

    def estimate(self, labeled):
        h = self.base
        recs = records(labeled)
        if not recs:
            return h
        groups = {}
        for bid, sk, _, y, s, _ in recs:
            g = groups.setdefault((bid, sk), [0, 0, sk, s])
            g[0] += y
            g[1] += 1
        K = np.array([g[0] for g in groups.values()], float)
        N = np.array([g[1] for g in groups.values()], float)
        M = np.array([self.ref.theta_prior(g[2], g[3])[0] for g in groups.values()])
        c = 1.0 / math.sqrt(1.0 + math.pi / 8.0 * (h.sigma_d ** 2 + h.sigma_g ** 2))
        S0 = math.sqrt(h.sigma_mu ** 2 + h.sigma_theta ** 2 + h.sigma_delta ** 2)
        mus = h.mu0 + self.tau_mu * np.linspace(-4, 4, 81)
        lss = math.log(S0) + self.tau_s * np.linspace(-4, 4, 41)
        eta = c * (mus[:, None, None, None] + M[None, None, :, None]
                   + np.exp(lss)[None, :, None, None] * _GH_X[None, None, None, :])
        ll = K[None, None, :, None] * -np.logaddexp(0, -eta) \
            + (N - K)[None, None, :, None] * -np.logaddexp(0, eta)
        mx = ll.max(-1, keepdims=True)
        lp = (mx[..., 0] + np.log(np.exp(ll - mx) @ _GH_W)).sum(-1)
        lp = lp - 0.5 * ((mus[:, None] - h.mu0) / self.tau_mu) ** 2 \
            - 0.5 * ((lss[None, :] - math.log(S0)) / self.tau_s) ** 2
        i, j = np.unravel_index(int(np.argmax(lp)), lp.shape)
        kw = {}
        if "c" in self.mode:
            kw["mu0"] = float(mus[i])
        if "s" in self.mode:
            S = math.exp(float(lss[j]))
            kw["sigma_mu"] = math.sqrt(max(S ** 2 - h.sigma_theta ** 2 - h.sigma_delta ** 2,
                                           EB_FLOOR ** 2))
        return replace(h, **kw)

    def predict(self, input, labeled=None):
        key = (id(labeled), len(labeled or ()))
        m = self.models.get(key)
        if m is None:
            try:
                hy = self.estimate(labeled)
            except Exception:
                self.failures += 1
                hy = self.base
            m = self.models[key] = HierPredictor(self.prior, hy)
            self.trace.append((len(labeled or ()), hy.mu0, hy.sigma_mu))
        return m.predict(input, labeled)


def hier_model(name, parent):
    nu, ov, eb = parse(name)
    prior, hyper = bundle(parent, nu)
    if ov is not None:
        hyper = replace(hyper, **ov)
    if eb:
        return EBHier(prior, hyper, eb)
    return HierPredictor(prior, hyper)


def factory(names, run, made):
    """() -> predict for one candidate family: `names` is [SMOOTHED], P_NAMES
    (the Predictor grid, a vector per call) or one hier config."""
    ids = T.anon_parents(run)
    if names == [SMOOTHED]:
        return lambda: B.smoothed_mean(4.0, 0.5)
    if names in (P_NAMES, P_EXT_NAMES, P_ROB_NAMES):
        variants = {0: P_VARIANTS, 1: P_EXT, 2: P_ROB}[[P_NAMES, P_EXT_NAMES, P_ROB_NAMES].index(names)]
        robust = names == P_ROB_NAMES

        def f():
            ms = {par: PredictorGrid(*ppred(par), variants, robust) for par in set(ids.values())}
            made.extend(ms.values())
            return lambda input, labeled=None: ms[ids[input[1]["benchmark_id"]]].predict(input, labeled)
        return f
    (name,) = names

    def f():
        ms = {par: hier_model(name, par) for par in set(ids.values())}
        made.extend(ms.values())
        return lambda input, labeled=None: ms[ids[input[1]["benchmark_id"]]].predict(input, labeled)
    return f


# --- harness -------------------------------------------------------------------------

def checkpoints(run, scope="pair", seed=0):
    """The run as paiec.official plays it with the platform's random policy:
    (slots, [(budget, labeled, inputs, index)]), each checkpoint's inputs
    deduplicated as official._evaluate does."""
    slots = O._slots(run, seed, scope)
    hook = O.make_hook(None, False)
    rt = O._Runtime(None, False, None)
    cps = []
    for b in BUDGETS:
        if b:
            O._acquire(slots, b, hook, rt)
        labeled = [e for s in slots for e in s.acquired[:b]]
        inputs, where, index = [], {}, []
        for s in slots:
            for inp, _, key in s.targets:
                j = where.setdefault(key, len(inputs))
                if j == len(inputs):
                    inputs.append(inp)
                index.append(j)
        cps.append((b, labeled, inputs, np.array(index)))
    return slots, cps


def evaluate(slots, cps, names, fac):
    """Score one family on every checkpoint: a fresh instance per checkpoint,
    one worker. -> ({name: per pair {'brier': [6], 'ece': [6], 'q0': mean B0
    prediction}}, timing)."""
    k = len(names)
    preds = {}
    times = []
    for b, labeled, inputs, index in cps:
        fn = fac()
        P = np.empty((len(inputs), k))
        for j, inp in enumerate(inputs):
            t = time.perf_counter()
            v = fn(inp, labeled)
            times.append(time.perf_counter() - t)
            v = np.atleast_1d(np.asarray(v, float))
            if v.shape != (k,) or not np.all(np.isfinite(v)) or np.any(v < 0) or np.any(v > 1):
                raise ValueError(f"invalid prediction {v!r}")
            P[j] = v
        preds[b] = P[index]
    out = {n: [] for n in names}
    at = 0
    for s in slots:
        y = np.array([t[1] for t in s.targets], float)
        sl = slice(at, at + len(y))
        at += len(y)
        for c, n in enumerate(names):
            br = [float(np.mean((preds[b][sl, c] - y) ** 2)) for b in BUDGETS]
            ec = [O.ece(list(preds[b][sl, c]), y) for b in BUDGETS]
            out[n].append({"brier": br, "ece": ec, "q0": float(np.mean(preds[0][sl, c]))})
    tm = {"calls": len(times), "mean_s": float(np.mean(times)), "max_s": float(np.max(times))}
    return out, tm


def alc(v):
    return float(np.dot(WEIGHTS, v))


def compact(rows, full):
    """What is stored per config and run: per pair ALC; the run's budget means,
    B0 ECE (mean and max over pairs), ECE-ALC and mean B0 prediction; with
    `full` also per pair Brier by budget (the references and every config
    scored on its own, not the grids)."""
    br = np.array([r["brier"] for r in rows])
    ec = np.array([r["ece"] for r in rows])
    out = {"alc": [round(alc(b), 6) for b in br],
           "b": [round(float(x), 6) for x in br.mean(0)],
           "ece0": [round(float(ec[:, 0].mean()), 5), round(float(ec[:, 0].max()), 5)],
           "ece_alc": round(float(np.mean([alc(e) for e in ec])), 5),
           "q0": round(float(np.mean([r["q0"] for r in rows])), 5)}
    if full:
        out["pb"] = [[round(float(x), 5) for x in b] for b in br]
    return out


def describe(regime, run):
    if REGIMES[regime][0] == "testlike":
        return [{"bench": d["pseudo"], "parent": d["parent"], "subject": d["subject_id"],
                 "eval": d["eval_items"], "p": round(d["p"], 5), "logit": round(d["logit"], 4)}
                for d in T.describe(run)]
    out = []
    for p, items in run:
        _, ev = (set(k) & set(items) for k in O.split(p, 0, "pair"))
        by = p.by_item()
        ys = [x.label for k in ev for x in by[k]]
        out.append({"bench": p.benchmark_id, "parent": p.benchmark_id, "subject": p.subject_id,
                    "eval": len(ev), "p": round(float(np.mean(ys)), 5),
                    "logit": round(T.logit_rate(sum(ys), len(ys)), 4)})
    return out


def task_key(t):
    return "|".join(map(str, t))


def run_task(t):
    """t = (regime, run index, family), family 'base' (smoothed, the Predictor
    grid, hier default) or a tuple of hier config names; ('dense', bench,
    names) for a dense public run."""
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if t[0] == "dense":
            run = dense(t[1])
            fams = [P_ROB_NAMES if n == "prob" else [n] for n in t[2]]
        else:
            regime, i, fam = t
            run = draw(regime, i)
            fams = [[SMOOTHED], P_NAMES, [HIER]] if fam == "base" else \
                [P_EXT_NAMES] if fam == "pext" else [P_ROB_NAMES] if fam == "prob" else \
                [[n] for n in fam]
        single = t[0] == "dense" or (t[2] not in ("base", "pext", "prob") and len(t[2]) == 1)
        slots, cps = checkpoints(run)
        res, timing = {}, {}
        for names in fams:
            made = []
            out, tm = evaluate(slots, cps, names, factory(names, run, made))
            fl = int(sum(getattr(m, "failures", 0) for m in made))
            un = int(sum(getattr(m, "unconverged", 0) for m in made))
            for n in names:
                res[n] = compact(out[n], full=n in (SMOOTHED, PRED, HIER) or single)
            key = names[0] if len(names) == 1 else "P grid" if names == P_NAMES else \
                "P ext" if names == P_EXT_NAMES else "P robust"
            timing[key] = {**tm, "failures": fl, "unconverged": un}
            eb = [m.trace for m in made if isinstance(m, EBHier)]
            if eb:
                timing[key]["eb"] = [[round(x, 4) for x in tr] for trace in eb for tr in trace]
    return task_key(t), {"res": res, "timing": timing, "task_s": round(time.perf_counter() - t0, 3)}


def run_meta(regime, i):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return describe(regime, draw(regime, i))


# --- verification --------------------------------------------------------------------

def verify(n_tl=2, n_r1=1):
    """The harness against official.run_official, pair for pair, and the
    shipped Predictor against testlike_check's stored rows on the same runs."""
    out = []
    stored = None
    if os.path.exists(TESTLIKE_CHECK):
        with open(TESTLIKE_CHECK) as f:
            stored = json.load(f)["raw"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cases = [("tl", i) for i in range(n_tl)] + [("r1b", i) for i in range(n_r1)]
        for regime, i in cases:
            run = draw(regime, i)
            slots, cps = checkpoints(run)
            ids = T.anon_parents(run)
            for names in ([SMOOTHED], P_NAMES, [HIER], [gname(-1.5, 1.3, 0.75)]):
                made = []
                got, _ = evaluate(slots, cps, names, factory(names, run, made))
                n = PRED if names == P_NAMES else names[0]
                if names == P_NAMES:
                    fac = lambda: (lambda ms: lambda inp, lab=None: ms[ids[inp[1]["benchmark_id"]]]
                                   .predict(inp, lab))({par: Predictor(*ppred(par))
                                                        for par in set(ids.values())})
                else:
                    fac = factory(names, run, [])
                ref = O.run_official(run, fac, deepcopy=False)
                d = max(abs(a - b) for r, g in zip(ref["rows"], got[n])
                        for a, b in zip([r["brier"][b] for b in BUDGETS], g["brier"]))
                out.append({"regime": regime, "run": i, "config": n, "max_abs_diff": d})
            if regime == "tl" and stored is not None and f"check|default|{i}" in stored:
                rows = stored[f"check|default|{i}"]["rows"]
                d = max(abs(x - y) for r, g in zip(rows, got_p(slots, cps, run))
                        for x, y in zip(r["pred"][:6], g))
                out.append({"regime": regime, "run": i, "config": "Predictor vs testlike_check",
                            "max_abs_diff": d})
    return out


def got_p(slots, cps, run):
    got, _ = evaluate(slots, cps, P_NAMES, factory(P_NAMES, run, []))
    return [g["brier"] for g in got[PRED]]


# --- statistics ----------------------------------------------------------------------

def mean_se(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0]


def alc_of(r):
    return np.asarray(r["alc"], float)


def unpack(r):
    """compact()'s dict from a packed row: [per-pair ALC, budget means (both in
    units of 1e-5), [B0 ECE mean, B0 ECE max, ECE-ALC, mean B0 prediction] (in
    units of 1e-4), per-pair Brier by budget (1e-5; references only)]."""
    if isinstance(r, dict):
        return r
    out = {"alc": [x * 1e-5 for x in r[0]], "b": [x * 1e-5 for x in r[1]],
           "ece0": [r[2][0] * 1e-4, r[2][1] * 1e-4], "ece_alc": r[2][2] * 1e-4,
           "q0": r[2][3] * 1e-4}
    if len(r) > 3:
        out["pb"] = [[x * 1e-5 for x in b] for b in r[3]]
    return out


def compact_raw(raw):
    """Pack stored rows in place (unpack reads them): integers in units of 1e-5
    (ALC, Brier) and 1e-4 (ECE, B0 prediction), positional; per-pair Brier by
    budget kept only for the references and dense runs, the summary reads it
    for none; timings rounded. Idempotent. It moves no summary number by more
    than the rounding (5e-6 on a mean ALC)."""
    for k, v in raw.items():
        if k.startswith("meta|"):
            continue
        for n, r in list(v["res"].items()):
            if isinstance(r, dict):
                # an earlier packing (u = 5) already held ALC in units of 1e-5
                scale = 1 if r.get("u") == 5 else 1e5
                row = [[int(round(x * scale)) for x in r["alc"]],
                       [int(round(x * 1e5)) for x in r["b"]],
                       [int(round(r["ece0"][0] * 1e4)), int(round(r["ece0"][1] * 1e4)),
                        int(round(r["ece_alc"] * 1e4)), int(round(r["q0"] * 1e4))]]
                if "pb" in r and (n in (SMOOTHED, PRED, HIER) or k.startswith("dense")):
                    row.append([[int(round(x * 1e5)) for x in b] for b in r["pb"]])
                v["res"][n] = row
        for tm in v["timing"].values():
            tm["mean_s"] = float(f"{tm['mean_s']:.4g}")
            tm["max_s"] = round(tm["max_s"], 4)


class Boot:
    """Cluster bootstrap over the clusters that appear in a set of runs (a ratio
    estimator: each pair appearance contributes x / run size and 1 / run size,
    so a resampled cluster brings its weight); W resamples all clusters as one
    pool, Ws within each parent benchmark."""

    def __init__(self, metas, boots=BOOTS, seed=0):
        self.keys = sorted({(r["parent"], r["subject"]) for m in metas for r in m})
        at = {k: j for j, k in enumerate(self.keys)}
        self.idx = [np.array([at[(r["parent"], r["subject"])] for r in m]) for m in metas]
        K = len(self.keys)
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(K, np.full(K, 1 / K), size=boots).astype(float)
        par = np.array([k[0] for k in self.keys])
        rs = np.random.default_rng([seed, 2])
        self.Ws = np.zeros_like(self.W)
        for p in sorted(set(par)):
            sel = np.flatnonzero(par == p)
            self.Ws[:, sel] = rs.multinomial(len(sel), np.full(len(sel), 1 / len(sel)), size=boots)
        self.n = np.zeros(K)
        for i in self.idx:
            np.add.at(self.n, i, 1.0 / len(i))

    def se(self, values):
        v = np.zeros(len(self.keys))
        for i, x in zip(self.idx, values):
            np.add.at(v, i, np.asarray(x, float) / len(i))
        return [float(((W @ v) / (W @ self.n)).std(ddof=1)) for W in (self.W, self.Ws)]


def lopo(metas, values):
    """Mean over runs of the run's pair-mean with one parent's pairs left out,
    per parent: the range says how much the number rests on one parent."""
    parents = sorted({r["parent"] for m in metas for r in m})
    out = {}
    for q in parents:
        xs = [np.mean([x for r, x in zip(m, v) if r["parent"] != q])
              for m, v in zip(metas, values) if any(r["parent"] != q for r in m)]
        out[q] = float(np.mean(xs))
    return out


def rows_of(raw, regime, i, name):
    """A config's stored rows for one run, as compact() writes them (floats),
    whether or not compact_raw has packed them since."""
    for fam in families_for(name):
        v = raw.get(task_key((regime, i, fam)))
        if v is not None and name in v["res"]:
            return unpack(v["res"][name])
    return None


def families_for(name):
    """The task families a config may be stored under."""
    if name in (SMOOTHED, HIER) or name in P_NAMES:
        return ["base"]
    if name in P_EXT_NAMES:
        return ["pext"]
    if name in P_ROB_NAMES:
        return ["prob"]
    return [c for c in _chunks_with(name)]


_CHUNKS = {}


def _chunks_with(name):
    return _CHUNKS.get(name, []) + [(name,)]


def register_chunks(chunks):
    for ch in chunks:
        for n in ch:
            _CHUNKS.setdefault(n, [])
            if tuple(ch) not in _CHUNKS[n]:
                _CHUNKS[n].append(tuple(ch))


def compare(raw, metas, regime, ids, names, ref=PRED, boot=None):
    """Per config on runs `ids`: ALC, budget means, ECE, B0 prediction, and the
    paired difference against `ref` with SEs; configs missing a run are
    skipped."""
    have = {n: [rows_of(raw, regime, i, n) for i in ids] for n in names + [ref]}
    base = have[ref]
    if any(x is None for x in base):
        return {}
    ms = [metas[(regime, i)] for i in ids]
    boot = boot or Boot(ms)
    out = {}
    for n in names:
        rs = have[n]
        if any(x is None for x in rs):
            continue
        alcs = [float(np.mean(alc_of(r))) for r in rs]
        diff = [alc_of(r) - alc_of(b) for r, b in zip(rs, base)]
        d_run = [float(np.mean(d)) for d in diff]
        se_c, se_s = boot.se(diff)
        out[n] = {"runs": len(ids), "ALC": mean_se(alcs),
                  "budgets": [float(x) for x in np.mean([r["b"] for r in rs], 0)],
                  "ece_alc": float(np.mean([r["ece_alc"] for r in rs])),
                  "ece0": float(np.mean([r["ece0"][0] for r in rs])),
                  "ece0_max": float(np.max([r["ece0"][1] for r in rs])),
                  "q0": float(np.mean([r["q0"] for r in rs])),
                  "diff": {"mean": float(np.mean(d_run)), "run_se": mean_se(d_run)[1],
                           "cluster_se": se_c, "strat_se": se_s,
                           "lopo": lopo(ms, diff)},
                  "diff_budgets": [float(x) for x in np.mean([np.array(r["b"]) - np.array(b["b"])
                                                              for r, b in zip(rs, base)], 0)]}
    return out


# --- stages --------------------------------------------------------------------------

def grid_chunks():
    """The Gaussian grids in tasks of one (attr_scale, sigma_mu) row of mu0s."""
    return [tuple(gname(m, s, a) for m in GRID_MU0) for a in GRID_AS for s in GRID_SM] + \
        [tuple(gname(m, s, a) for m in GRID2_MU0) for a in GRID2_AS for s in GRID2_SM]


def t3_points(raw, n_sel):
    """Student-t levels at the Gaussian grids' two best (mu0, attr_scale) on
    the selection half, at two scales (the t scale is about 0.7 of the
    Gaussian's; each costs about seven Gaussian fits)."""
    best = ranked(raw, [n for n in GRID_NAMES], range(n_sel))
    pts, seen = [], set()
    for n in best:
        _, ov, _ = parse(n)
        k = (ov["mu0"], ov["attr_scale"])
        if k in seen:
            continue
        seen.add(k)
        pts.append(k)
        if len(pts) == 2:
            break
    return [gname(m, s, a, nu=3) for m, a in pts for s in (1.3, 1.8)]


def eb_configs(raw, n_sel):
    """EB on the fitted defaults and on the best Gaussian config of the
    selection half: scale only, centre and scale, and centre and scale with a
    wider prior on the run's centre (TAU_MU 2); then, added once the guard
    was seen to bind, centre and scale on moderately shifted Gaussian bases
    (the shift the guard allows at B0, EB adapting from B1)."""
    top = ranked(raw, GRID_NAMES, range(n_sel))
    out = [ebname("s", HIER), ebname("cs", HIER), ebname("cs,tm=2.0", HIER)]
    if top:
        out += [ebname("s", top[0]), ebname("cs", top[0]), ebname("cs,tm=2.0", top[0])]
    out += [ebname("cs,tm=2.0", gname(m, 2.5, a)) for a in (0.5, 1.0)
            for m in (-3.0, -2.5, -2.0, -1.5)]
    out += [ebname("cs", gname(m, 2.5, 0.5)) for m in (-2.5, -2.0)]
    return list(dict.fromkeys(out))


def ranked(raw, names, ids, regime="tl"):
    """names with every run in `ids` stored, best mean ALC first."""
    score = {}
    for n in names:
        rs = [rows_of(raw, regime, i, n) for i in ids]
        if rs and all(r is not None for r in rs):
            score[n] = float(np.mean([np.mean(alc_of(r)) for r in rs]))
    return sorted(score, key=score.get)


def screen_configs(raw, n_sel):
    """hier configs screened against the guard on public runs (benchmark-first
    runs 0..SCREEN_RUNS-1): a frontier over the level shift (mu0, attr_scale)
    at sigma_mu 2.5 (the best width at every shift on the test-like selection
    half), within the test-like grids, and every EB config."""
    g = [gname(m, 2.5, a) for a in (0.5, 1.0) for m in (-4.0, -3.0, -2.5, -2.0, -1.5, -1.0)]
    g += [gname(m, 2.5, 0.25) for m in (-5.0, -4.0, -3.5, -3.0)]
    g += [gname(m, 2.5, 0.75) for m in (-2.5, -2.0, -1.5, -1.0)]
    return g + eb_configs(raw, n_sel)


def screen_loss(raw, name):
    """Mean ALC difference against the shipped Predictor on the screen runs, or
    None when a run is missing."""
    d = []
    for i in range(SCREEN_RUNS):
        r, b = rows_of(raw, "r1b", i, name), rows_of(raw, "r1b", i, PRED)
        if r is None or b is None:
            return None
        d.append(float(np.mean(alc_of(r)) - np.mean(alc_of(b))))
    return float(np.mean(d))


def shortlist(raw, n_sel):
    """The hier configs carried to confirmation: the SHORT_G best on the
    selection half among the screened ones (Gaussian and EB) that lose at most
    GUARD on the screen, then for reference the two best Gaussian configs and
    the best EB config on the fitted defaults whatever they lose, and the
    best Student-t config (test-like only: its guard is not measured, so it
    cannot be chosen). Every Predictor variant is scored everywhere anyway."""
    ids = range(n_sel)
    scr = screen_configs(raw, n_sel)
    loss = {n: screen_loss(raw, n) for n in scr}
    passing = [n for n in ranked(raw, scr, ids) if loss[n] is not None and loss[n] <= GUARD]
    g = ranked(raw, GRID_NAMES, ids)
    t = ranked(raw, t3_points(raw, n_sel), ids)
    eb0 = ranked(raw, [n for n in eb_configs(raw, n_sel) if n.endswith("@" + HIER)], ids)
    hier = list(dict.fromkeys(passing[:SHORT_G] + g[:2] + eb0[:1] + t[:1]))
    # the sensitivity regimes and dense runs: the best passing config, the best
    # one that passed with a margin (no loss on the screen), and the best EB
    # on the fitted defaults (no bet on the level at B0)
    margin = [n for n in passing if loss[n] <= 0]
    sens = list(dict.fromkeys(passing[:1] + margin[:1] + eb0[:1]))
    return {"hier": hier, "passing": passing, "sens": sens,
            "screen": {n: v for n, v in loss.items() if v is not None}}


def tasks_for(stage, raw, n):
    half = n // 2
    sel, conf = range(half), range(half, n)
    if stage == "grid":
        chunks = grid_chunks()[:len(GRID_AS) * len(GRID_SM)]
        return [("tl", i, "base") for i in sel] + [("tl", i, ch) for i in sel for ch in chunks]
    if stage == "grid2":
        chunks = grid_chunks()[len(GRID_AS) * len(GRID_SM):]
        return [("tl", i, "pext") for i in sel] + [("tl", i, ch) for i in sel for ch in chunks]
    if stage == "prob":
        return [("tl", i, "prob") for i in sel]
    if stage == "screen":
        return [("r1b", i, (c,)) for i in range(SCREEN_RUNS) for c in screen_configs(raw, half)]
    if stage == "r1base":
        return [(r, i, f) for r in ("r1b", "r1p") for i in range(R1_RUNS)
                for f in ("base", "pext", "prob")]
    if stage == "t3":
        return [("tl", i, (c,)) for i in sel for c in t3_points(raw, half)]
    if stage == "eb":
        return [("tl", i, (c,)) for i in sel for c in eb_configs(raw, half)]
    if stage == "confirm":
        sl = shortlist(raw, half)
        top = sl["hier"]
        guarded = [c for c in top if parse(c)[0] == 0]
        out = [("tl", i, f) for i in conf for f in ("base", "pext", "prob")]
        out += [("tl", i, (c,)) for i in conf for c in top]
        for r in ("r1b", "r1p"):
            out += [(r, i, f) for i in range(R1_RUNS) for f in ("base", "pext", "prob")]
            out += [(r, i, (c,)) for i in range(R1_RUNS) for c in guarded]
        sens = sl["sens"]
        for r in SENS:
            out += [(r, i, f) for i in range(SENS_RUNS) for f in ("base", "prob")]
            out += [(r, i, (c,)) for i in range(SENS_RUNS) for c in sens]
        out += [("dense", b, ("prob", HIER) + tuple(sl["sens"][:2])) for b in DENSE]
        return out
    if stage == "final":
        # the chosen config and the best guarded hier configs of each family, on
        # the sensitivity regimes and dense runs where confirmation had not
        # scored them
        sel = summarise({"raw": raw, "config": {"runs": n}})["selected"]
        picks = [sel["chosen"]] + [v for f, v in sel["best_guarded_per_family"].items()
                                   if f.startswith("hier")]
        picks = [c for c in dict.fromkeys(picks) if c and c.startswith("hier") and c != HIER]
        out = [(r, i, (c,)) for r in SENS for i in range(SENS_RUNS) for c in picks]
        return out + [("dense", b, tuple(picks)) for b in DENSE]
    raise ValueError(stage)


R1_RUNS, SENS_RUNS, SCREEN_RUNS = 100, 100, 40
#: What docs/findings.md ("Calibrating for the hidden test") recommends
#: shipping. The rule's argmax (select) is mu0 -3.5; -3.0 is 0.0004 behind it
#: on the selection half, level with it on the confirmation half, and 0.0017
#: (paired SE 0.0003) cheaper on both public weightings; 'ship' in the summary
#: pairs it with the alternatives.
SHIP = "hier G mu0=-3.00 sm=2.50 as=0.25"


# --- summary -------------------------------------------------------------------------

def collect_metas(raw):
    metas = {}
    for k, v in raw.items():
        if k.startswith("meta|"):
            _, regime, i = k.split("|")
            metas[(regime, int(i))] = v
    return metas


def summarise(state):
    raw, cfg = state["raw"], state["config"]
    n = cfg["runs"]
    half = n // 2
    register_chunks(grid_chunks())
    metas = collect_metas(raw)
    sl = shortlist(raw, half)
    cand = [SMOOTHED, PRED, HIER] + sl["hier"] + P_ALL
    cand = list(dict.fromkeys(cand))
    out = {"shortlist": sl, "feedback": {"budgets": list(map(float, np.mean(
        [r[2] for r in T.FEEDBACK], 0))), "ALC": T.FEEDBACK_ALC}}

    # selection half: every config
    ids_sel = [i for i in range(half) if ("tl", i) in metas]
    boot = Boot([metas[("tl", i)] for i in ids_sel])
    every = [SMOOTHED, HIER] + P_ALL + GRID_NAMES + t3_points(raw, half) + eb_configs(raw, half)
    every = list(dict.fromkeys(every))
    sel = compare(raw, metas, "tl", ids_sel, every, boot=boot)
    out["selection"] = {"runs": len(ids_sel), "configs": sel}

    ids_conf = [i for i in range(half, n) if ("tl", i) in metas]
    ids_all = [i for i in range(n) if ("tl", i) in metas]
    out["confirmation"] = compare(raw, metas, "tl", ids_conf, cand) if ids_conf else {}
    out["testlike_all"] = compare(raw, metas, "tl", ids_all, cand) if ids_conf else {}
    guard = {}
    for r in ("r1b", "r1p"):
        ids = [i for i in range(R1_RUNS) if (r, i) in metas]
        guard[r] = compare(raw, metas, r, ids, cand) if ids else {}
    out["r1"] = guard
    out["sensitivity"] = {}
    for r in SENS:
        ids = [i for i in range(SENS_RUNS) if (r, i) in metas]
        if ids:
            out["sensitivity"][r] = compare(raw, metas, r, ids, cand)
    out["selected"] = select(out, cand)
    out["chosen_vs"] = chosen_vs(raw, metas, out, n)
    out["ship"] = {"config": SHIP, "vs": chosen_vs(raw, metas, out, n, SHIP)}
    out["fit_ab"] = {"check": fitab_check(), "shipped_divergences": divergences(raw, metas)}
    out["ppc"] = ppc(out, raw, n)
    out["latency"] = latency(raw)
    out["eb_adaptation"] = eb_adaptation(raw, metas)
    out["grid_surface"] = grid_surface(sel)
    return out


def select(out, cand):
    """The best config on the selection half whose loss against the shipped
    Predictor on public runs is at most GUARD on both weightings; its
    confirmation-half difference and the optimism (selection - confirmation)."""
    sel = out["selection"]["configs"]
    order = sorted((n for n in cand if n in sel), key=lambda n: sel[n]["ALC"][0])
    rows = []
    chosen = None
    for n in order:
        g = [out["r1"].get(r, {}).get(n, {}).get("diff", {}).get("mean") for r in ("r1b", "r1p")]
        ok = all(x is not None and x <= GUARD for x in g)
        c = out["confirmation"].get(n, {}).get("diff", {})
        rows.append({"config": n, "selection_ALC": sel[n]["ALC"][0],
                     "selection_diff": sel[n]["diff"]["mean"],
                     "confirmation_diff": c.get("mean"), "r1_diffs": g, "guard_ok": ok})
        if ok and chosen is None:
            chosen = n
    best_family = {}
    for r in rows:
        f = family(r["config"])
        if f not in best_family and r["guard_ok"]:
            best_family[f] = r["config"]
    res = {"order": rows, "chosen": chosen, "best_guarded_per_family": best_family}
    if chosen:
        s = sel[chosen]["diff"]["mean"]
        c = out["confirmation"].get(chosen, {}).get("diff", {}).get("mean")
        res["optimism"] = None if c is None else s - c
    return res


NEAR = 0.25


def fitab_check(n=3000, seed=0):
    """fitting.fit_ab against robust_fit_ab: one test-like pair where fit_ab
    diverges (27 of 31, no item difficulties, prior mean -1.883, va 3), and
    n random small problems (up to 31 labels, z ~ N(0, 1), a pair rate drawn
    uniformly, prior mean N(0, 1.5^2), va 2 or 3): how often fit_ab misses the
    MAP, by how much in log posterior, what that does to the prediction at z = 0,
    and how far the two differ where they agree."""
    def logpost(a, b, Z, Y, m, va):
        eta = a + b * Z
        return float(-np.sum(Y * np.logaddexp(0, -eta) + (1 - Y) * np.logaddexp(0, eta))
                     - (a - m) ** 2 / (2 * va) - (b - 1) ** 2 / (2 * 0.25))

    def pred(r):
        return float(sig(r[0] / math.sqrt(1 + math.pi / 8 * r[2])))

    Z, Y = np.zeros(31), np.array([1.0] * 27 + [0.0] * 4)
    ex = {"fit_ab": [round(x, 4) for x in fit_ab(Z, Y, -1.883, 3.0, 1.0, 0.25)],
          "robust": [round(x, 4) for x in robust_fit_ab(Z, Y, -1.883, 3.0, 1.0, 0.25)]}
    rng = np.random.default_rng(seed)
    gaps, dp, agree = [], [], 0.0
    for _ in range(n):
        k = int(rng.integers(0, 32))
        Z = rng.normal(0, 1, k)
        Y = (rng.random(k) < rng.random()).astype(float)
        m, va = float(rng.normal(0, 1.5)), float(rng.choice([2.0, 3.0]))
        a, b = fit_ab(Z, Y, m, va, 1.0, 0.25), robust_fit_ab(Z, Y, m, va, 1.0, 0.25)
        d = max(abs(x - y) for x, y in zip(a, b))
        if d > 1e-6:
            gaps.append(logpost(b[0], b[1], Z, Y, m, va) - logpost(a[0], a[1], Z, Y, m, va))
            dp.append(abs(pred(a) - pred(b)))
        else:
            agree = max(agree, d)
    return {"example": ex, "problems": n, "missed": len(gaps),
            "logpost_gain": [float(np.min(gaps)), float(np.median(gaps))] if gaps else None,
            "pred_diff": [float(np.median(dp)), float(np.max(dp))] if dp else None,
            "max_diff_where_agreeing": agree}


def divergences(raw, metas):
    """Pair appearances where the shipped Predictor's ALC differs from the same
    Predictor with robust_fit_ab ('Predictor rf') by more than 1e-4, per
    regime: fit_ab diverged on some checkpoint of that pair."""
    out = {}
    for regime in sorted({r for r, _ in metas}):
        n, bad = 0, []
        for (r, i) in metas:
            if r != regime:
                continue
            a, b = rows_of(raw, r, i, PRED), rows_of(raw, r, i, PRED + " rf")
            if a is None or b is None:
                continue
            d = alc_of(a) - alc_of(b)
            n += len(d)
            bad += [float(x) for x in d if abs(x) > 1e-4]
        out[regime] = {"pairs": n, "diverged": len(bad), "alc_excess": sorted(bad)}
    return out


def chosen_vs(raw, metas, out, n, chosen=None):
    """A config (default the rule's choice) against the best guarded config of
    every family and the two best guarded runners-up, paired, on the test-like
    halves and the public runs (the difference config minus other, with its
    SEs)."""
    sel = out["selected"]
    chosen = chosen or sel.get("chosen")
    if not chosen:
        return {}
    others = list(sel["best_guarded_per_family"].values())
    runner = [r["config"] for r in sel["order"] if r["guard_ok"] and r["config"] != chosen]
    others = [o for o in dict.fromkeys([sel.get("chosen")] + others + runner[:2])
              if o and o != chosen]
    half = n // 2
    sets = {"tl confirmation": ("tl", range(half, n)), "tl all": ("tl", range(n)),
            "r1b": ("r1b", range(R1_RUNS)), "r1p": ("r1p", range(R1_RUNS))}
    res = {}
    for o in others:
        res[o] = {}
        for label, (regime, ids) in sets.items():
            ids = [i for i in ids if (regime, i) in metas]
            got = compare(raw, metas, regime, ids, [chosen], ref=o) if ids else {}
            res[o][label] = got.get(chosen, {}).get("diff")
    return res


def ppc(out, raw, n_runs):
    """Per candidate on all test-like runs: budget means against the feedback,
    and the feedback shifted by the paired difference against the Predictor,
    the candidate's estimate on the real run (the Predictor's own row is the
    feedback). 'near' repeats the difference on the NEAR share of runs whose
    Predictor budget profile is closest to the feedback's (each budget over
    its sd across runs), in case the difference depends on the run's shape.
    The Predictor's own z-scores against the feedback come first."""
    fb = np.array(out["feedback"]["budgets"])
    ids = [i for i in range(n_runs) if rows_of(raw, "tl", i, PRED) is not None]
    pb = np.array([rows_of(raw, "tl", i, PRED)["b"] for i in ids])
    sd = pb.std(0, ddof=1)
    dist = np.sqrt((((pb - fb) / sd) ** 2).sum(1))
    near = [ids[j] for j in np.argsort(dist)[:max(1, int(round(NEAR * len(ids))))]]
    res = {"_predictor_z": {"mean": [float(x) for x in pb.mean(0)], "sd": [float(x) for x in sd],
                            "z_feedback": [float(x) for x in (fb - pb.mean(0)) / sd],
                            "near_runs": len(near),
                            "near_mean": [float(x) for x in pb[np.argsort(dist)[:len(near)]].mean(0)]}}
    for n, v in out.get("testlike_all", {}).items():
        est = fb + np.array(v["diff_budgets"])
        rs = [(rows_of(raw, "tl", i, n), rows_of(raw, "tl", i, PRED)) for i in near]
        dn = np.mean([np.array(r["b"]) - np.array(b["b"]) for r, b in rs], 0) \
            if all(r is not None for r, _ in rs) else None
        res[n] = {"budgets": v["budgets"], "ALC": v["ALC"][0],
                  "feedback_plus_diff": [float(x) for x in est],
                  "feedback_plus_diff_ALC": float(T.FEEDBACK_ALC + v["diff"]["mean"]),
                  "near_feedback_plus_diff": None if dn is None else [float(x) for x in fb + dn],
                  "near_feedback_plus_diff_ALC": None if dn is None else
                  float(T.FEEDBACK_ALC + np.dot(WEIGHTS, dn)),
                  "q0": v["q0"], "ece0": v["ece0"], "ece0_max": v["ece0_max"]}
    return res


def latency(raw):
    """Evaluation call wall-clock per config: mean over runs of the per-run mean,
    the largest single call, per regime kind; dense runs apart."""
    acc = defaultdict(lambda: {"mean": [], "max": 0.0, "runs": 0, "task_s": []})
    for k, v in raw.items():
        if k.startswith("meta|"):
            continue
        kind = "dense " + k.split("|")[1] if k.startswith("dense") else \
            ("r1" if k.startswith("r1") else "testlike")
        for n, tm in v["timing"].items():
            a = acc[(kind, n)]
            a["mean"].append(tm["mean_s"])
            a["max"] = max(a["max"], tm["max_s"])
            a["runs"] += 1
            if k.startswith("dense"):
                a["task_s"].append(v["task_s"])
    return {f"{k[0]} | {k[1]}": {"runs": a["runs"], "mean_ms": 1e3 * float(np.mean(a["mean"])),
                                 "max_s": a["max"],
                                 **({"task_s": a["task_s"]} if a["task_s"] else {})}
            for k, a in sorted(acc.items())}


def eb_adaptation(raw, metas):
    """EB estimates per config and regime, by budget: mean mu0 and sigma_mu over
    the per-parent models of every run (a run's checkpoints are told apart by
    their label counts, which grow with the budget), next to the regime's mean
    pair logit on the evaluated responses (accuracy scale)."""
    acc = defaultdict(list)
    for k, v in raw.items():
        if k.startswith("meta|") or k.startswith("dense"):
            continue
        regime = k.split("|")[0]
        for n, tm in v["timing"].items():
            tr = tm.get("eb", [])
            counts = sorted({int(x[0]) for x in tr})
            if len(counts) != len(BUDGETS):
                continue
            at = {c: b for c, b in zip(counts, BUDGETS)}
            for nl, mu, sm in tr:
                acc[(n, regime, at[int(nl)])].append((mu, sm))
    out = defaultdict(dict)
    for (n, regime, b), xs in sorted(acc.items()):
        a = np.array(xs)
        out[f"{n} | {regime}"][f"B{b}"] = [float(a[:, 0].mean()), float(a[:, 1].mean()), len(xs)]
    levels = defaultdict(list)
    for (regime, _), m in metas.items():
        levels[regime] += [r["logit"] for r in m]
    return {"estimates": dict(out),
            "pair_logit_mean": {r: float(np.mean(v)) for r, v in levels.items()}}


ALL_MU0 = sorted(set(GRID_MU0) | set(GRID2_MU0))
ALL_SM = sorted(set(GRID_SM) | set(GRID2_SM))
ALL_AS = sorted(set(GRID_AS) | set(GRID2_AS))


def grid_surface(sel):
    """Selection-half ALC of the Gaussian grids as [attr_scale][sigma_mu][mu0]
    over ALL_MU0 (None where not scored)."""
    return {"mu0": ALL_MU0,
            **{f"as={a}": {f"sm={s}": [sel.get(gname(m, s, a), {}).get("ALC", [None])[0]
                                       for m in ALL_MU0] for s in ALL_SM} for a in ALL_AS}}


# --- driver --------------------------------------------------------------------------

def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, separators=(",", ":"))
    os.replace(tmp, path)


def provenance():
    files = ["paiec/hier.py", "paiec/prior.py", "paiec/predict.py", "paiec/official.py",
             "paiec/testlike.py", "experiments/level_calibration.py"]
    dig = {}
    for f in files:
        with open(os.path.join(ROOT, f), "rb") as fh:
            dig[f] = hashlib.sha256(fh.read()).hexdigest()[:16]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "paiec"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        head, dirty = None, None
    return {"head": head, "paiec_dirty": dirty, "digests": dig}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["grid", "grid2", "prob", "r1base", "t3", "eb", "screen",
                                        "confirm", "final"])
    ap.add_argument("--runs", type=int, default=200, help="test-like runs, halves select/confirm")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="run at most this many tasks")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if args.verify:
        for r in verify():
            print(r)
        return
    state = load(args.out) if (args.resume or args.summarise) else None
    if state is None:
        state = {"status": "partial", "config": {"runs": args.runs, "boots": BOOTS,
                                                 "guard": GUARD, "tau_mu": TAU_MU,
                                                 "tau_s": TAU_S, "eb_floor": EB_FLOOR,
                                                 "r1_runs": R1_RUNS, "sens_runs": SENS_RUNS,
                                                 "regimes": {k: [v[0], v[1], v[2]]
                                                             for k, v in REGIMES.items()},
                                                 "p_grid": P_VARIANTS, "grid": [GRID_MU0, GRID_SM,
                                                                               GRID_AS]},
                 "passes": [], "raw": {}}
    raw = state["raw"]
    register_chunks(grid_chunks())
    if args.summarise:
        compact_raw(raw)
        state["summary"] = summarise(state)
        state["status"] = "complete" if state["summary"].get("r1") else "partial"
        save(args.out, state)
        print(json.dumps(state["summary"]["selected"], indent=1)[:4000])
        return
    if not args.stage:
        ap.error("--stage, --summarise or --verify")
    t0 = time.time()
    tasks = [t for t in tasks_for(args.stage, raw, state["config"]["runs"])
             if task_key(t) not in raw]
    if args.limit:
        tasks = tasks[:args.limit]
    # run descriptors (cheap, in the parent)
    need = sorted({(t[0], t[1]) for t in tasks if t[0] != "dense"
                   and f"meta|{t[0]}|{t[1]}" not in raw})
    print(f"stage {args.stage}: {len(tasks)} tasks, {len(need)} run descriptors", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for regime, i in need:
            raw[f"meta|{regime}|{i}"] = run_meta(regime, i)
        parents = sorted({p.benchmark_id for p in pairs()})
        bundles = {(par, nu): bundle(par, nu) for par in parents for nu in (0.0, 3.0)}
        ppriors = {par: ppred(par) for par in parents}
    save(args.out, state)
    done, last = 0, time.time()
    # longest first, so a slow task does not finish the stage alone
    order = sorted(tasks, key=lambda t: (t[0] != "dense", not (isinstance(t[2], tuple) and any(
        parse(n)[0] > 0 for n in t[2] if n.startswith("hier"))), str(t)))
    with mp.get_context("fork").Pool(args.jobs, initializer=init,
                                      initargs=(bundles, ppriors)) as pool:
        for key, val in pool.imap_unordered(run_task, order):
            raw[key] = val
            done += 1
            if time.time() - last > 60 or done == len(order):
                save(args.out, state)
                last = time.time()
                print(f"  {done}/{len(order)} tasks, {time.time() - t0:.0f}s", flush=True)
    state["passes"].append({"stage": args.stage, "command": " ".join(sys.argv),
                            "tasks": len(order), "wall_s": round(time.time() - t0, 1),
                            "jobs": args.jobs, **provenance()})
    save(args.out, state)


if __name__ == "__main__":
    main()
