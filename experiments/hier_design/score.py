"""Exact scoring of count-based predictors on the R1 runs (runs.pkl), and a
pair-level generative simulator for scenarios the public data do not cover.

A count-based predictor here is predict(own_k, own_n, others) -> probability,
where `others` lists (k, n) of the other pairs of the same benchmark in the run
at the same budget: everything a label-count model can read from the shared
`labeled` list about a target's level.
"""
import pickle

from common import *  # noqa: F401,F403
from paiec.evaluator import BUDGETS, WEIGHTS

RUNS = os.path.join(HERE, "runs.pkl")


def load_runs():
    with open(RUNS, "rb") as f:
        return pickle.load(f)


def score_run(run, predictor, bench_of=lambda p: p["bench"]):
    """Brier by budget (pair-averaged) and ALC of one run, exactly as the replica
    scores a predictor that is constant within a pair."""
    by = {}
    for j, p in enumerate(run):
        by.setdefault(bench_of(p), []).append(j)
    out = np.zeros(len(BUDGETS))
    for bi, B in enumerate(BUDGETS):
        tot = 0.0
        for j, p in enumerate(run):
            k, n = sum(p["labels"][:B]), min(B, len(p["labels"]))
            others = [(sum(run[o]["labels"][:B]), min(B, len(run[o]["labels"])))
                      for o in by[bench_of(p)] if o != j]
            f = predictor(k, n, others, p)
            pe = p["p_eval"]
            tot += pe * (1 - pe) + (f - pe) ** 2
        out[bi] = tot / len(run)
    return out, float(np.dot(WEIGHTS, out))


def score_runs(runs, predictor, **kw):
    res = [score_run(r, predictor, **kw) for r in runs]
    by_budget = np.array([r[0] for r in res])
    alc = np.array([r[1] for r in res])
    return by_budget, alc


def smoothed(n0=4.0, c=0.5):
    return lambda k, n, others, p: (k + n0 * c) / (n + n0)


# --- logit-normal hierarchical predictor on a grid --------------------------

class Hier:
    """eta = m + u, m ~ prior_m (benchmark level), u ~ prior_u (pair deviation),
    p = sigmoid(eta). Own labels inform m + u, other pairs of the benchmark
    inform m. Posterior mean of p by quadrature on a (m, u) grid. The priors
    are normal, or Student-t with `df` for heavier tails. use_others=False
    ignores the other pairs (a single prior on eta with sd sqrt(sm^2+su^2))."""

    def __init__(self, mu0, sm, su, df_m=None, df_u=None, use_others=True, nm=121, nu=121):
        from scipy import stats
        self.m = mu0 + sm * np.linspace(-7, 7, nm) if sm > 0 else np.array([mu0])
        self.u = su * np.linspace(-7, 7, nu)
        dm = stats.norm if df_m is None else stats.t(df_m)
        du = stats.norm if df_u is None else stats.t(df_u)
        self.wm = dm.pdf((self.m - mu0) / sm) if sm > 0 else np.ones(1)
        self.wm = self.wm / self.wm.sum()
        self.wu = du.pdf(self.u / su)
        self.wu = self.wu / self.wu.sum()
        self.P = sig(self.m[:, None] + self.u[None, :])          # (nm, nu)
        self.lP, self.l1P = np.log(self.P), np.log1p(-self.P)
        self.use_others = use_others
        self._cache = {}

    def _pair_lik(self, k, n):
        """likelihood over the (m, u) grid of k successes in n."""
        return np.exp(k * self.lP + (n - k) * self.l1P)

    def _other_lik(self, k, n):
        key = ("o", k, n)
        if key not in self._cache:
            self._cache[key] = self._pair_lik(k, n) @ self.wu              # over m
        return self._cache[key]

    def __call__(self, k, n, others, p=None):
        key = (k, n, tuple(sorted(others)) if self.use_others else ())
        if key in self._cache:
            return self._cache[key]
        wm = self.wm.copy()
        if self.use_others:
            for ko, no in others:
                if no:
                    wm = wm * self._other_lik(ko, no)
                    wm = wm / wm.sum()
        W = wm[:, None] * self.wu[None, :] * self._pair_lik(k, n)
        f = float((W * self.P).sum() / W.sum())
        self._cache[key] = f
        return f


# --- scenarios -------------------------------------------------------------------

def simulate(runs, M, S_mu, S_u, rng, df_m=None, df_u=None, n_labels=31):
    """Replace every run's pairs by draws from the hierarchical model, keeping
    the run's structure (which pairs share a benchmark): a fresh level per
    benchmark per run, a fresh deviation per pair, Bernoulli labels, and the
    evaluation rate set to the pair's p (so the Brier constant is p(1-p))."""
    out = []
    for run in runs:
        lv = {}
        new = []
        for p in run:
            b = p["bench"]
            if b not in lv:
                z = rng.standard_t(df_m) if df_m else rng.standard_normal()
                lv[b] = M + S_mu * z
            z = rng.standard_t(df_u) if df_u else rng.standard_normal()
            pr = float(sig(lv[b] + S_u * z))
            new.append({"bench": b, "labels": (rng.random(n_labels) < pr).astype(int).tolist(),
                        "p_eval": pr})
        out.append(new)
    return out
