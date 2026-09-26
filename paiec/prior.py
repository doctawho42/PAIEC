"""Offline prior for paiec.hier, and its hyperparameters by empirical Bayes.

Both are fitted on public pairs minus an exclusion set of benchmark names, so an
experiment can leave the run's benchmarks out of everything it learns, the
honest stand-in for a hidden test whose benchmarks were never public.

One item-level model per public benchmark (benchmark_fit) is the source of
everything: logit p = mu_b + t_s - g_i - e_i on each subject's first recorded
response per item (the one acquisition reveals), with the item residual
integrated as at run time and the variances found by Laplace-EM. It gives
  * each pair's standing t_s, centred within its benchmark: the target of the
    attribute ridge and of the identity table, on the run-time model's scale
    (the Predictor's prior is on the accuracy-logit scale, about half as wide);
  * the benchmark levels, item variance and group share that fit_hyper pools.
A benchmark's fit does not depend on which others are excluded, so fits are
cached per benchmark and leave-one-benchmark-out costs one fit per benchmark.

The run-time half (canon_name, SubjectPrior's lookups and JSON round trip)
imports numpy only; pandas is imported by the offline half, which alone needs it.
main() reads the public data, so if this module ships, the build must treat it
as offline only.

Every hyperparameter the included benchmarks cannot identify is set to
REFERENCE, which no fit went into: `python -m paiec.prior` prints which fields
are estimated, which fell back and which rest on a stand-in, on all five public
benchmarks and leaving each out; its first row is paiec.hier.Hyper's defaults.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from itertools import combinations

import numpy as np

from .subjects import Spec, _field, attrs, design_row

PROVIDERS = {"alibaba", "amazon", "anthropic", "bytedance", "cohere", "deepseek",
             "google", "meta", "minimax", "mistral", "moonshot", "nvidia", "openai",
             "stanford", "stepfun", "xai", "zhipu"}
#: widening of the public ML sd of (level, pair deviation) toward the Gaussian
#: widths that stayed robust in the step-2 width sweep (1.3 / 0.9, 1.3 / 0.93)
WIDEN = (1.44, 1.4)
#: largest group share of the item variance: above 0.3 benchmarks whose key is
#: useless lost up to 0.0118 ALC in the step-2 sweep
G_CAP = 0.25
#: What fit_hyper takes for a field the included benchmarks cannot identify.
#: No fit went into these, so a leave-benchmarks-out fit never inherits a
#: number fitted on the benchmarks it left out (Hyper's defaults are such
#: numbers). mu0 = 0 is the one that matters: a new benchmark sits at p = 0.5
#: before attributes, and B0 is scored on it; the step-2 review found the
#: public mean level worse than 0.5 at B0 once it was fitted without the
#: scored benchmark, and much better when it was not. sigma_mu = 2.5 is
#: the conventional weakly informative scale of a logistic intercept. It is
#: not widened like an estimate: WIDEN corrects an ML sd fitted on the same
#: few levels, and the 1 + 1/n term the uncertainty of their mean, and 2.5
#: about 0 is already 1.27 times the public levels' root mean square about 0
#: (1.97). The other widths are round Rasch-scale values: 2.5 for an item and
#: for a pair's own standing, which on this scale ranges as widely as the
#: items (models that solve nothing to models that solve everything), 1 for
#: the group and attribute terms, and theta at 0.3 so that linking stays weak
#: (weight 0.014 with an attribute prior, 0.15 without). sigma_delta was 1.0
#: until the final review: every public estimate lies between 1.6 and 3.2,
#: and on runs simulated from the default model 1.0 cost 0.0021 ALC
#: (final_verify/c1_calib.py); 2.5 was set knowing that range, so it is the
#: one REFERENCE value chosen with the public data in view.
REFERENCE = dict(mu0=0.0, sigma_mu=2.5, sigma_theta=0.3, sigma_delta=2.5, sigma_attr=1.0,
                 sigma_d=2.5, sigma_g=1.0)
#: fewest benchmark levels behind an estimated mu0 or sigma_mu. With one or
#: two, their mean is the level of the very benchmarks a strict run-LOBO fit
#: left, known with no spread: a run holding only multi_swebench's other
#: benchmarks got mu0 = -4.14, which cost 0.013 ALC on runs simulated from the
#: default model (final review, final_verify/c1_calib.py)
MIN_LEVELS = 3


NAME_CHARS = 512
_PAREN = re.compile(r"\([^()\n]*\)")


def canon_name(subject) -> str:
    """normalized_name, else source_model_name from subject_features_extra (47% of
    matharena subjects have no name), lower-cased, parentheticals such as
    '(Think)' or '(high)' and provider words dropped, '-_/' as spaces.

    The only key that links a model across public benchmarks: harness,
    reasoning_effort, access_date and subject_features_extra are formats of one
    benchmark, so the full dict links none of matharena's or multi_swebench's.
    The name is cut at NAME_CHARS and a parenthetical may not hold a paren:
    the lazy '\\(.*?\\)' it replaces was quadratic on a string of '(' (2.5 s
    at 20k), and every public name is unchanged (tests/test_hier.py).
    """
    try:
        name = _field(subject, "normalized_name").strip()
        if not name:
            m = re.search(r"source_model_name=([^;]{1,%d})" % NAME_CHARS,
                          _field(subject, "subject_features_extra")[:4 * NAME_CHARS])
            name = m[1] if m else ""
        t = name[:NAME_CHARS].lower()
        while True:         # innermost parentheticals first; linear per pass
            u = _PAREN.sub(" ", t)
            if u == t:
                break
            t = u
        t = re.sub(r"[-_/]", " ", t)
        return " ".join(w for w in t.split() if w not in PROVIDERS)
    except Exception:
        return ""


class SubjectPrior:
    """Where a subject stands within a benchmark before any label, on the
    run-time model's scale.

    attribute(subject) -> (mean, u): the ridge over paiec.subjects' design, as
        fit_prior builds it (alpha 2, no intercept), fitted on the standings;
        u = x' C x is the uncertainty of the ridge's coefficients at x.
    identity(subject) -> (mean, v) or None: the same canonical name's standing
        on the public benchmarks it was seen on, as an attribute residual from
        ridges that saw neither the benchmark nor the name (_residuals), or
        raw when attributes are off, benchmark offsets removed, with noise
        v = s2d / k over the k benchmarks that mean covers. paiec.hier
        precision-weights it against theta's prior and caps the weight: after
        attributes no transfer is measurable (tau2_res -0.002 of a residual
        variance of 3.09 on the public data, a name-bootstrap sd of 0.44).

    table rows are [raw mean, residual mean or None, benchmarks behind the raw
    mean, benchmarks behind the residual one]. from_dict validates shapes and
    finiteness and raises ValueError, so a broken prior.json fails when it is
    loaded (or packaged) rather than turning every prediction into a fallback.
    """

    def __init__(self, coef=None, spec=None, cov=None, table=None, s2d=None, meta=None):
        self.coef = None if coef is None else np.asarray(coef, float)
        self.spec = spec
        self.cov = None if cov is None else np.asarray(cov, float)
        self.table = dict(table or {})
        self.s2d = dict(s2d or {})
        self.meta = dict(meta or {})

    @property
    def has_attributes(self):
        return self.coef is not None and self.spec is not None

    def attribute(self, subject):
        x = design_row(attrs(subject), self.spec)
        m = float(x @ self.coef)
        u = float(x @ self.cov @ x) if self.cov is not None else 0.0
        if not (math.isfinite(m) and math.isfinite(u)):
            return 0.0, 0.0
        return m, max(u, 0.0)

    def identity(self, subject, residual=True):
        key = canon_name(subject)
        row = self.table.get(key) if key else None
        s2d = self.s2d.get("resid" if residual else "raw")
        if row is None or s2d is None or not math.isfinite(s2d) or s2d <= 0:
            return None
        mean = row[1] if residual else row[0]
        if mean is None or not math.isfinite(mean):
            return None
        k = row[3] if residual and len(row) > 3 else row[2]
        return float(mean), float(s2d) / max(k, 1)

    def validate(self):
        """Raise ValueError unless every number is finite and every shape is
        what the run-time lookups index."""
        def finite(x, what):
            a = np.asarray(x, float)
            if not np.all(np.isfinite(a)):
                raise ValueError(f"subject prior: non-finite {what}")
            return a
        if (self.coef is None) != (self.spec is None):
            raise ValueError("subject prior: coef and spec come together")
        if self.coef is not None:
            n = len(design_row(attrs({}), self.spec))
            if finite(self.coef, "coef").shape != (n,):
                raise ValueError(f"subject prior: {self.coef.shape} coefficients, "
                                 f"the design has {n}")
            if self.cov is not None and finite(self.cov, "cov").shape != (n, n):
                raise ValueError(f"subject prior: cov {self.cov.shape}, need ({n}, {n})")
            finite([self.spec.med_days, self.spec.med_log_size], "spec medians")
        for k, row in self.table.items():
            if not isinstance(row, (list, tuple)) or len(row) not in (3, 4) \
                    or row[1] is not None and not math.isfinite(float(row[1])):
                raise ValueError(f"subject prior: table row {k!r} is malformed")
            finite(row[0], f"table row {k!r}")
            counts = finite(row[2:], f"table row {k!r}")
            if any(int(n) != n or n < 0 for n in counts) or counts[0] < 1:
                raise ValueError(f"subject prior: table row {k!r} counts are malformed")
        for k, v in self.s2d.items():
            if v is not None:
                finite(v, f"s2d {k}")
        return self

    def to_dict(self):
        return {"coef": None if self.coef is None else self.coef.tolist(),
                "spec": None if self.spec is None else self.spec.to_dict(),
                "cov": None if self.cov is None else self.cov.tolist(),
                "table": {k: list(v) for k, v in self.table.items()},
                "s2d": self.s2d, "meta": self.meta}

    @classmethod
    def from_dict(cls, d):
        try:
            return cls(d.get("coef"), Spec.from_dict(d["spec"]) if d.get("spec") else None,
                       d.get("cov"), d.get("table"), d.get("s2d"), d.get("meta")).validate()
        except (TypeError, KeyError, AttributeError, IndexError) as err:
            raise ValueError(f"subject prior: {err!r}") from err

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


# --- one public benchmark ------------------------------------------------------------

_FITS: dict = {}


def benchmark_fit(pairs_b, s2e=None, max_iter=500, tol=1e-4):
    """Item-level model of one benchmark: level, standings, variances.

    logit p = mu + t_s - g_i - e_i with t_s ~ N(0, s2_t) (one per pair: within a
    benchmark a subject is a pair), g_i the group effects of the run-time key
    rule with a variance per key, e_i ~ N(0, s2e) integrated per item. Laplace-EM:
    x = (mu, t, u) at its mode given the variances (paiec.hier.Problem), each
    variance updated to the mean posterior second moment of its effects, the
    item residual's from its exact per-item posterior. With a single subject
    the item variance is not identified, so `s2e` must then be given; mu gets a
    flat N(0, 100) prior. Cached on content: the pairs in order (t follows
    it), every (subject, item, label) cell, each item's features and
    interactors, and s2e; a count summary served a stale fit to labels that
    changed with their counts (step-2 review).
    """
    from .hier import Problem, _inverse, item_groups, select_keys, MISSING
    bid = pairs_b[0].benchmark_id
    items, index, si, ji, y = [], {}, [], [], []
    digest = hashlib.blake2b(digest_size=16)
    for s, p in enumerate(pairs_b):
        digest.update(f"\x1e{p.subject_id}".encode("utf-8", "surrogatepass"))
        for k, rs in p.by_item().items():
            if k not in index:
                index[k] = len(items)
                items.append(rs[0].item)
                digest.update("\x1d{}\x1f{}\x1f{}".format(
                    k, rs[0].item.get("item_features"),
                    rs[0].item.get("interactors")).encode("utf-8", "surrogatepass"))
            si.append(s)
            ji.append(index[k])
            y.append(float(rs[0].label))
            digest.update(f"\x1c{index[k]}:{int(rs[0].label)}".encode())
    key = (bid, digest.hexdigest(), None if s2e is None else round(float(s2e), 6))
    if key in _FITS:
        return _FITS[key]
    si, ji, y = np.array(si), np.array(ji), np.array(y)
    n_s, n_i = len(pairs_b), len(items)
    free_t = n_s >= 2
    if not free_t and s2e is None:
        raise ValueError(f"{bid}: one subject, so the item variance must be given")
    feats = [item_groups(it) for it in items]
    keys = select_keys(feats)
    gcol, gkey = {}, []
    base = 1 + (n_s if free_t else 0)
    gcols = []
    for f in feats:
        row = []
        for k in keys:
            g = (k, f.get(k, MISSING))
            if g not in gcol:
                gcol[g] = base + len(gcol)
                gkey.append(k)
            row.append(gcol[g])
        gcols.append(row)
    p = base + len(gcol)
    S = 1 + int(free_t) + len(keys)
    cols = np.zeros((len(y), S), np.int64)
    vals = np.zeros((len(y), S))
    vals[:, 0] = 1.0
    if free_t:
        cols[:, 1] = 1 + si
        vals[:, 1] = 1.0
    if keys:
        cols[:, 1 + int(free_t):] = np.array(gcols)[ji]
        vals[:, 1 + int(free_t):] = -1.0
    s2t, s2g = 3.0, {k: 1.0 for k in keys}
    s2 = 6.0 if s2e is None else float(s2e)
    gk = np.array(gkey)
    m = np.zeros(p)

    def variances():
        V = np.full(p, 100.0)
        if free_t:
            V[1:1 + n_s] = s2t
        for k in keys:
            V[base:][gk == k] = s2g[k]
        return V

    prob = Problem(cols, vals, y, ji, np.full(n_i, s2), m, variances())
    x, e0, done, it = None, None, False, 0
    for it in range(1, max_iter + 1):
        st, P = prob.solve(x, e0)
        x, e0 = st.x, st.it.mode
        d = np.diag(_inverse(P))
        old = [s2t, s2] + [s2g[k] for k in keys]
        if free_t:
            s2t = float(np.mean(x[1:1 + n_s] ** 2 + d[1:1 + n_s]))
        for k in keys:
            sel = base + np.flatnonzero(gk == k)
            s2g[k] = float(np.mean(x[sel] ** 2 + d[sel]))
        if s2e is None:
            s2 = float(np.mean(st.it.ebar ** 2 + st.it.evar))
        s2t, s2 = max(s2t, 1e-3), max(s2, 1e-3)
        s2g = {k: max(v, 1e-3) for k, v in s2g.items()}
        new = [s2t, s2] + [s2g[k] for k in keys]
        prob.V, prob.s2e = variances(), np.full(n_i, s2)
        # a variance on its way to zero crawls under EM and matters to nothing
        if all(abs(math.log(a / b)) < tol or max(a, b) < 0.01 for a, b in zip(new, old)):
            done = True
            break
    st, P = prob.solve(x, e0)
    C = _inverse(P)
    t = st.x[1:1 + n_s] if free_t else np.zeros(n_s)
    tv = np.diag(C)[1:1 + n_s] if free_t else np.zeros(n_s)
    gsum = float(sum(s2g.values()))
    out = {"benchmark": bid, "n_pairs": n_s, "n_items": n_i, "n_cells": len(y),
           "mu": float(st.x[0]), "mu_var": float(C[0, 0]),
           "t": t.tolist(), "t_var": tv.tolist(), "subjects": [p.subject for p in pairs_b],
           "s2_t": s2t if free_t else None, "s2e": s2, "s2_g": s2g, "keys": keys,
           "item_var": s2 + gsum, "share": gsum / (s2 + gsum) if keys else None,
           "iterations": it, "converged": done, "newton_converged": bool(prob.converged)}
    _FITS[key] = out
    return out


def _by_benchmark(pairs, exclude=()):
    held = set(exclude)
    by = defaultdict(list)
    for p in pairs:
        if p.benchmark_id not in held:
            by[p.benchmark_id].append(p)
    return dict(sorted(by.items()))


# --- the subject prior ---------------------------------------------------------------

def standings(pairs, exclude=()):
    """[(benchmark, subject dict, centred standing, its posterior variance)] over
    the included benchmarks with at least two pairs; a lone pair has no
    relative standing."""
    rows = []
    for b, ps in _by_benchmark(pairs, exclude).items():
        if len(ps) < 2:
            continue
        f = benchmark_fit(ps)
        t = np.array(f["t"])
        for p, z, v in zip(ps, t - t.mean(), f["t_var"]):
            rows.append((b, p.subject, float(z), float(v)))
    return rows


def _ridge(rows, alpha):
    """(coef, spec, X, (X'X + alpha I)^-1) on rows: the spec (provider and
    effort columns, imputation medians) is fixed by the rows' attributes."""
    import pandas as pd
    import warnings
    df = pd.DataFrame([attrs(s) for _, s, _, _ in rows])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spec = Spec.from_frame(df)
    # no subject with a date or a size leaves a nan median, and nan everywhere
    spec.med_days = spec.med_days if math.isfinite(spec.med_days) else 0.0
    spec.med_log_size = spec.med_log_size if math.isfinite(spec.med_log_size) else 0.0
    X = np.vstack([design_row(a, spec) for a in df.to_dict("records")])
    z = np.array([r[2] for r in rows])
    A = X.T @ X + alpha * np.eye(X.shape[1])
    Ai = np.linalg.inv(A)
    return Ai @ X.T @ z, spec, X, Ai


#: ridge penalty of the identity pass (_residuals for tau2_res): least squares
#: in effect. At the run-time alpha the two ridges behind a name's residuals
#: share their shrinkage toward 0 and tau2 comes out high (from_standings)
ID_ALPHA = 1e-6


def _residuals(rows, alpha, X=None):
    """(resid, weights): each standing's attribute residual from a ridge that
    saw neither its benchmark nor any row of its model; all nan with fewer
    than two benchmarks.

    Every ridge is fitted on X, the design of all rows under the full ridge's
    spec (default: _ridge(rows)'s), which is what the run-time prior uses and
    holds no standing. A spec rebuilt on each training subset dropped the
    columns with fewer than 8 rows left in it (37 of the 67 public ridges:
    effort 'high' in 19, meta in 18, bytedance in 15), and a dropped column's
    effect went into both of a name's residuals: with the shrinkage at alpha
    2 (from_standings), +0.14 on tau2 = 0 drawn on the public design (step-2b
    review, verify_id/synth_tau.py).

    Row i on benchmark b gets z_i less the prediction of a ridge fitted
    without b, centred by that ridge's mean prediction over b's rows (z is
    centred within b). A canonical name with rows on other benchmarks too
    gets, on each of its benchmarks, a ridge of its own that also leaves out
    every row of the name (21 names on the public data, 67 ridges in all).
    Without that, the ridge that predicts the name on b had fitted its
    standing on the others, which pulls its residuals on two benchmarks
    apart: their mean cross-product estimated tau2 - 2 h Var(z) for rows of
    leverage h (-0.24 on the public data). Each ridge is centred on its own
    mean: centring on the benchmark ridge's mean, which has seen the name,
    lets it back in. Rows without a canonical name link nothing. A row with
    no training row left keeps nan.

    weights[i], for the rows of a linked name: the prediction's centred
    smoother weights over all rows (w' z is its centred prediction, zero
    where the ridge did not train)."""
    n = len(rows)
    z = np.array([r[2] for r in rows])
    bench = np.array([r[0] for r in rows])
    names = np.array([canon_name(r[1]) for r in rows], dtype=object)
    resid, weights = np.full(n, np.nan), {}
    benches = sorted(set(bench))
    if len(benches) < 2:
        return resid, weights
    X = _ridge(rows, alpha)[2] if X is None else np.asarray(X, float)
    eye = alpha * np.eye(X.shape[1])
    seen = defaultdict(set)
    for b, k in zip(bench, names):
        if k:
            seen[k].add(b)
    for b in benches:
        te = np.flatnonzero(bench == b)
        D = X[te]
        linked = sorted({names[i] for i in te if len(seen.get(names[i], ())) >= 2})
        for k in [None] + linked:
            keep = bench != b if k is None else (bench != b) & (names != k)
            own = np.ones(len(te), bool) if k is None else names[te] == k
            tr = np.flatnonzero(keep)
            if not len(tr):
                resid[te[own]] = np.nan
                continue
            Xt = X[tr]
            Ai = np.linalg.inv(Xt.T @ Xt + eye)
            pred = D @ (Ai @ (Xt.T @ z[tr]))
            resid[te[own]] = z[te[own]] - (pred[own] - pred.mean())
            if k is not None:
                W = D @ Ai @ Xt.T
                Wc = W[own] - W.mean(0)
                for i, w in zip(te[own], Wc):
                    full = np.zeros(n)
                    full[tr] = w
                    weights[int(i)] = full
    return resid, weights


def _link_noise(members, weights, group, sigma2):
    """noise(k, b1, b2) -> (c0, c1): what the two ridges' errors add to the
    cross-product of name k's residuals on b1 and b2 is c0 + c1 tau2,
    averaged over the rows of the two cells.

    A ridge's centred prediction is w'z (_residuals), so the two errors
    covary by w1' Cov(z) w2 about the attribute function. Rows of one name
    (`group`: one id per name, and per row for rows without one) share tau2,
    every row has sigma2 in all, so Cov(z) = (sigma2 - tau2) I + tau2 S with
    S the same-name indicator: c0 = sigma2 w1'w2 and c1 = W1'W2 - w1'w2, W
    the weights summed per name. On the public design (from_standings) the
    least-squares pass loses 0.08 to it at tau2 = 0 and 0.15 at 0.3, where
    without the name term (c1 = 0) the estimate was 0.363 +- 0.022 against
    0.308 +- 0.019 with it (200 sets, step-2b fix_id/nameterm.py). This
    takes out the ridges' noise, not a bias they share (from_standings)."""
    n_groups = int(group.max()) + 1 if len(group) else 0
    summed = {i: np.bincount(group, w, n_groups) for i, w in weights.items()}

    def noise(k, b1, b2):
        a = [(weights[i] @ weights[j], summed[i] @ summed[j])
             for i in members[(k, b1)] for j in members[(k, b2)]]
        ww, WW = np.mean(a, axis=0)
        return sigma2 * float(ww), float(WW - ww)
    return noise


def _offsets(cells):
    """Benchmark offsets c_b in u_kb = a_k + c_b (sum c_b = 0), from names seen on
    two or more benchmarks: centred standing shifts with a benchmark's pool
    (matharena's is 2026-heavy), not with the model."""
    by_k = defaultdict(dict)
    for (k, b), u in cells.items():
        by_k[k][b] = u
    multi = {k: d for k, d in by_k.items() if len(d) >= 2}
    bs = sorted({b for d in multi.values() for b in d})
    c = dict.fromkeys(bs, 0.0)
    if len(bs) < 2:
        return c
    for _ in range(500):
        a = {k: np.mean([u - c[b] for b, u in d.items()]) for k, d in multi.items()}
        new = {b: np.mean([d[b] - a[k] for k, d in multi.items() if b in d]) for b in bs}
        mean = np.mean(list(new.values()))
        new = {b: v - mean for b, v in new.items()}
        if max(abs(new[b] - c[b]) for b in bs) < 1e-10:
            return new
        c = new
    return c


def _components(cells, noise=None):
    """(tau2, s2d, subtracted) on offset-corrected per-(name, benchmark) means:
    s2d the name's variance across benchmarks, tau2 the part shared across
    them, the mean cross-product of one name's values on two benchmarks less
    what noise(k, b1, b2) = (c0, c1) says the rest adds to it, c0 + c1 tau2,
    averaged over the same cross-products: tau2 = (xp - c0) / (1 + c1).
    `subtracted` is xp - tau2 (0 without `noise`). (None, None, None) without
    a name on two benchmarks."""
    by_k = defaultdict(dict)
    for (k, b), u in cells.items():
        by_k[k][b] = u
    multi = {k: d for k, d in by_k.items() if len(d) >= 2}
    if not multi:
        return None, None, None
    mu = float(np.mean([u for d in multi.values() for u in d.values()]))
    xp, c = [], []
    for k, d in multi.items():
        for b1, b2 in combinations(sorted(d), 2):
            xp.append((d[b1] - mu) * (d[b2] - mu))
            c.append((0.0, 0.0) if noise is None else noise(k, b1, b2))
    ss = sum(float(((u - u.mean()) ** 2).sum())
             for u in (np.array(list(d.values())) for d in multi.values()))
    dof = sum(len(d) - 1 for d in multi.values())
    c0, c1 = np.mean(c, axis=0)
    tau2 = (float(np.mean(xp)) - c0) / max(1.0 + c1, 0.5)
    return float(tau2), ss / dof, float(np.mean(xp) - tau2)


def build_prior(pairs, exclude=(), alpha=2.0):
    """SubjectPrior from the included benchmarks, or None when no benchmark with
    two pairs is left: from_standings on standings(pairs, exclude), with
    meta['included'] every benchmark of `pairs` outside `exclude`, what
    paiec.hier.HierPredictor compares with the hyperparameters'
    Hyper.included, so a prior built on pairs filtered beforehand (exclude
    empty) cannot meet hyperparameters fitted on more."""
    rows = standings(pairs, exclude)
    if not rows:
        return None
    return from_standings(rows, alpha, sorted(_by_benchmark(pairs, exclude)), exclude)


def from_standings(rows, alpha=2.0, included=None, excluded=()):
    """SubjectPrior from standings rows [(benchmark, subject, standing,
    posterior variance)]. meta carries what fit_hyper needs: the pooled
    within-benchmark variance of the standings, the attribute residual
    variance, the mean coefficient uncertainty, and the identity components.
    meta['benchmarks'] are the benchmarks behind the rows; `included`
    (default: the same) and `excluded` are recorded for the checks
    fit_hyper and paiec.hier make.

    One residual per row serves the identity table, s2d_res and s2_res:
    _residuals at the run-time alpha, from a ridge on the full ridge's design
    that saw neither the row's benchmark nor its model, which is what a new
    subject on a new benchmark meets. s2_res is the variance a new subject's
    standing has about its attribute prediction (3.09 on the public data),
    s2d_res a name's spread across benchmarks (3.00), the table's residual
    means their average.

    tau2_res, the part of that variance a name shares across benchmarks,
    comes from a second pass of the same ridges at ID_ALPHA, least squares
    in effect: the mean cross-product of a name's residuals on two
    benchmarks, less what the two ridges' noise adds to it (_link_noise).
    What the two ridges get wrong in common, the noise correction cannot
    see, and the cross-product counts it as shared standing. At the run-time
    alpha that is mostly their shrinkage toward 0: on standings drawn on the
    public design (its 220 rows' names, benchmarks and attributes, f the
    full ridge's fitted values, a pair deviation of sd 1, a known tau2; 200
    sets each, step-2b fix_id/synth_shipped.py) the alpha-2 pass gave
    +0.077 +- 0.009 for tau2 = 0, 0.165 for 0.09 and 0.367 +- 0.018 for 0.3,
    the least-squares pass +0.019 +- 0.008, 0.105 +- 0.011 and 0.308 +- 0.019
    at the same spread (per-set sd 0.12 to 0.26). Subtracting the shrinkage
    cross-product at the fitted coefficients instead took off 0.024 of the
    passes' difference (fix_id/proto.py): those coefficients are shrunk
    themselves. Computed exactly, the residuals' systematic part
    cross-multiplies to 0.082 at alpha 2, 0.034 of it shrinkage, and to
    0.027 at least squares (fix_id/decomp2.py). That remainder is the one
    intercept: the ridge fits standings centred within benchmarks whose
    pools differ, and the offsets it cannot fit bend its slopes the same way
    for both residuals. It grows with the attribute effects (+0.058 at twice
    the fitted coefficients) and with effects outside the design (+0.079
    for one such term; fix_id/shipped_bigmis.txt), so the estimate errs
    high. A design demeaned within each training benchmark removed it in
    simulation but took the public estimate to -0.33, below every other
    variant, so it is not used. On the public data tau2_res is -0.002
    (0.097 at alpha 2, kept in meta['tau2_res_alpha']), -0.51 to +0.08
    leaving one benchmark out, with a cluster-bootstrap sd of 0.44 over the
    21 linked names (fix_id/boot_fixed.txt): nothing measurable is shared.

    The standings are posterior means, shrunk toward 0, so a true standing's
    variance is theirs plus the mean posterior variance (Laplace-EM's own
    fixed point for s2_t, e.g. matharena 7.39 = 7.21 + 0.18), and likewise for
    a residual: the noise is added, not subtracted."""
    coef, spec, X, Ai = _ridge(rows, alpha)
    z = np.array([r[2] for r in rows])
    noise = np.array([r[3] for r in rows])
    benches = sorted({r[0] for r in rows})
    resid, w_alpha = _residuals(rows, alpha, X)
    resid_id, weights = _residuals(rows, ID_ALPHA, X)
    ok = np.isfinite(resid)
    total = float(np.mean(z ** 2) + noise.mean())
    s2_res = float(np.mean(resid[ok] ** 2) + noise[ok].mean()) if ok.any() else None
    fitted = X @ coef
    sigma2 = s2_res if s2_res is not None else float(np.mean((z - fitted) ** 2) + noise.mean())
    cov = max(sigma2, 1e-6) * Ai @ X.T @ X @ Ai
    u = np.einsum("ij,jk,ik->i", X, cov, X)

    cells_raw, cells_res, cells_id, members = (defaultdict(list) for _ in range(4))
    group, ids = np.zeros(len(rows), np.int64), {}
    for i, (b, s, _, _) in enumerate(rows):
        k = canon_name(s)
        group[i] = ids.setdefault(k or ("row", i), len(ids))
        if k:
            cells_raw[(k, b)].append(z[i])
            if ok[i]:
                cells_res[(k, b)].append(resid[i])
                cells_id[(k, b)].append(resid_id[i])
                members[(k, b)].append(i)

    def centred(cells):
        cells = {kb: float(np.mean(v)) for kb, v in cells.items()}
        off = _offsets(cells)
        return {(k, b): v - off.get(b, 0.0) for (k, b), v in cells.items()}
    cells_raw, cells_res, cells_id = centred(cells_raw), centred(cells_res), centred(cells_id)
    tau_raw, s2d_raw, _ = _components(cells_raw)
    # what the ridges' errors are made of: the standings' own variance about
    # the attribute function, the residuals' less the prediction's (the
    # coefficients' covariance keeps s2_res, as before)
    s2z = max(float(np.mean(resid[ok] ** 2) - u[ok].mean()), 1e-6) if ok.any() else 0.0
    tau_alpha, s2d_res, _ = _components(cells_res, _link_noise(members, w_alpha, group, s2z))
    tau_res, _, link_noise = _components(cells_id, _link_noise(members, weights, group, s2z))
    table = defaultdict(lambda: [[], []])
    for (k, b), v in cells_raw.items():
        table[k][0].append(v)
    for (k, b), v in cells_res.items():
        table[k][1].append(v)
    table = {k: [float(np.mean(r)), float(np.mean(e)) if e else None, len(r), len(e)]
             for k, (r, e) in sorted(table.items())}
    meta = {"benchmarks": benches, "included": benches if included is None else sorted(included),
            "excluded": sorted(set(excluded)), "n_rows": len(rows),
            "total_var": total, "s2_res": s2_res, "u_mean": float(u.mean()),
            "tau2_raw": tau_raw, "tau2_res": tau_res, "link_noise": link_noise,
            "tau2_res_alpha": tau_alpha,
            "s2d_raw": s2d_raw, "s2d_res": s2d_res,
            "linked": sum(1 for r in table.values() if r[2] >= 2), "alpha": alpha}
    return SubjectPrior(coef, spec, cov, table, {"raw": s2d_raw, "resid": s2d_res}, meta)


# --- hyperparameters by empirical Bayes ---------------------------------------------

def fit_hyper(pairs, exclude=(), prior=None, widen=WIDEN, g_cap=G_CAP, nu_mu=0.0,
              centre="mean"):
    """paiec.hier.Hyper from the included benchmarks; (hyper, report).

    Each field is estimated when the included data can identify it and set to
    REFERENCE otherwise, never to a default fitted on other benchmarks;
    report['estimated'] and report['fallback'] name which is which, and
    report['approximate'] the estimates that rest on a stand-in (below).
      mu0, sigma_mu   from MIN_LEVELS = 3 benchmark levels or more, else both
                      REFERENCE (mu0 = 0): the levels' mean, and for a new
                      level about it the sd s widened to s sqrt(1 + 1/n), the
                      predictive spread of a new draw about the mean of n,
                      then times widen[0] for the Gaussian level; with
                      Student-t tails (nu_mu > 0) s sqrt(1 + 1/n) itself is
                      the scale (REFERENCE's 2.5 serves either way, unwidened,
                      see REFERENCE). A level is taken where the run-time eta
                      has it: at attribute score 0, i.e. the fitted mu plus
                      the pool's mean standing less its mean attribute score
                      under `prior` (a public pool's mean score runs from
                      -0.53 to +0.35 under the prior fitted on all five,
                      -0.88 to +0.55 under its own leave-out prior, so the
                      level at the average subject is not that).
                      centre='zero' keeps mu0 = 0 and takes the levels' root
                      mean square about 0 as their spread (no 1 + 1/n: no
                      mean is estimated), again from 3 levels: the choice for
                      when the public mean level is not to be trusted on a
                      new benchmark (to be settled by the evaluation)
      sigma_d/g       the median item variance of multi-subject benchmarks,
                      split by their median group share, capped at g_cap
      sigma_theta     identity tau2 of attribute residuals (from_standings),
                      clipped to [0.01, 0.2]: -0.002 on all five public
                      benchmarks, so sigma_theta 0.1 and a link weight of
                      0.0018; -0.51 to +0.08 leaving one out (sigma_theta 0.29
                      without matharena, 0.1 otherwise). Its noise is large (a
                      name-bootstrap sd of 0.44 on the public data, 0.12 on
                      synthetic standings a third as noisy) and its bias
                      positive (+0.02 at the public fit's coefficients, more
                      with larger or unmodelled effects). The floor keeps
                      theta proper and the link weight near 0.002 when noise
                      takes the estimate below zero, as in five of those six
                      fits. The ceiling is itself within noise of zero, and it
                      keeps one high draw from setting the link weight past
                      0.036 (sigma_delta 2.30 there), where the identity
                      weight has already reached id_cap on every public row
                      whose name has two residual benchmarks and on 73% of the
                      others (0.083 or more for the rest; at the floor 91% and
                      12%): above it only the link would move. Neither bound
                      is fitted: the evaluation found link weights up to 0.3
                      worth nothing measurable at formative size
                      (docs/findings.md)
      sigma_delta     LOBO attribute residual variance less theta's and the
                      coefficients' share, times widen[1]. With a single
                      multi-subject benchmark there is no LOBO residual: its
                      standings' whole variance stands in, as if the
                      attributes explained nothing on a new benchmark (LOBO
                      they explain about a quarter on public data, so this
                      errs wide), flagged in report['approximate']
      sigma_attr      what the attributes explain of the standings' variance
                      (LOBO only; REFERENCE with a single benchmark)
    slip, guess, text_share and id_cap are not estimated. `prior` defaults to
    build_prior(pairs, exclude); one built with another exclusion set, or on
    other pairs, is refused. The exclusion set is recorded in Hyper.excluded,
    the benchmarks fitted on in Hyper.included.

    Strict run-LOBO (every benchmark of a formative run left out) leaves on
    public data no benchmark in 48.5% of R1 runs (all REFERENCE), one
    multi-subject benchmark in 36.8% and only swe_rebench in 6.5%: a strict
    run-LOBO score of paiec.hier mostly measures these fallbacks. The
    target-LOBO line (exclude only the target's benchmark, one model per
    benchmark of the run, as experiments/official_baselines.per_target does for
    the Predictor) is the one that measures the model.
    """
    from .hier import Hyper
    if centre not in ("mean", "zero"):
        raise ValueError(f"centre must be 'mean' or 'zero', got {centre!r}")
    held = sorted(set(exclude))
    by = _by_benchmark(pairs, exclude)
    if prior is not None:
        if sorted(prior.meta.get("excluded", ())) != held:
            raise ValueError(f"prior excludes {prior.meta.get('excluded')}, fit_hyper {held}")
        inc = prior.meta.get("included")
        if inc is not None and sorted(inc) != sorted(by):
            raise ValueError(f"prior fitted on {sorted(inc)}, fit_hyper on {sorted(by)}")
    ref = dict(REFERENCE)
    multi = {b: benchmark_fit(ps) for b, ps in by.items() if len(ps) >= 2}
    est, rep = {}, {"benchmarks": list(by), "excluded": held, "centre": centre,
                    "approximate": []}
    if multi:
        T = float(np.median([f["item_var"] for f in multi.values()]))
        shares = [f["share"] for f in multi.values() if f["share"] is not None]
        share = min(float(np.median(shares)), g_cap) if shares else \
            ref["sigma_g"] ** 2 / (ref["sigma_g"] ** 2 + ref["sigma_d"] ** 2)
        est["sigma_g"], est["sigma_d"] = math.sqrt(share * T), math.sqrt((1 - share) * T)
        rep.update(item_var=T, share=share, shares=shares)
    else:
        T = ref["sigma_d"] ** 2 + ref["sigma_g"] ** 2
    fits = dict(multi)
    for b, ps in by.items():
        if b not in fits:
            fits[b] = benchmark_fit(ps, s2e=T)
    prior = build_prior(pairs, exclude) if prior is None else prior
    attr = prior is not None and prior.has_attributes
    levels = {}
    for b, ps in by.items():
        shift = float(np.mean(fits[b]["t"]))
        if attr:
            shift -= float(np.mean([prior.attribute(p.subject)[0] for p in ps]))
        levels[b] = fits[b]["mu"] + shift
    mus = np.array([levels[b] for b in by])
    n = len(mus)
    wide = widen[0] if nu_mu <= 0 else 1.0
    if centre == "zero":
        ref.pop("mu0")
        if n >= MIN_LEVELS:
            est["sigma_mu"] = float(np.sqrt(np.mean(mus ** 2))) * wide
    elif n >= MIN_LEVELS:
        est["mu0"] = float(np.mean(mus))
        est["sigma_mu"] = float(np.std(mus, ddof=1) * math.sqrt(1 + 1 / n)) * wide
    rep["levels"], rep["levels_fitted"] = levels, {b: fits[b]["mu"] for b in by}
    meta = prior.meta if prior is not None else {}
    th2 = ref["sigma_theta"] ** 2
    if meta.get("tau2_res") is not None:
        th2 = float(np.clip(meta["tau2_res"], 0.01, 0.2))
        est["sigma_theta"] = math.sqrt(th2)
    if meta.get("s2_res") is not None:
        est["sigma_delta"] = math.sqrt(max(meta["s2_res"] - meta["u_mean"] - th2, 0.1)) * widen[1]
        est["sigma_attr"] = math.sqrt(max(meta["total_var"] - meta["s2_res"], 0.0))
    elif meta.get("total_var") is not None:
        est["sigma_delta"] = math.sqrt(max(meta["total_var"] - meta["u_mean"] - th2, 0.1)) * widen[1]
        rep["approximate"].append("sigma_delta")
    fallback = {k: v for k, v in ref.items() if k not in est}
    rep["estimated"], rep["fallback"] = sorted(est), sorted(fallback)
    rep["prior_meta"] = meta
    mu0 = {"mu0": 0.0} if centre == "zero" else {}
    return Hyper(**{**Hyper().to_dict(), **fallback, **mu0, **est, "nu_mu": float(nu_mu),
                    "excluded": held, "included": sorted(by)}), rep


def build(pairs, exclude=(), nu_mu=0.0):
    """(SubjectPrior or None, Hyper) from every pair outside `exclude`."""
    prior = build_prior(pairs, exclude)
    hyper, _ = fit_hyper(pairs, exclude, prior, nu_mu=nu_mu)
    return prior, hyper


# --- prior.json ------------------------------------------------------------------------

KIND = "paiec.hier/1"


def to_json(prior, hyper) -> dict:
    """Everything the run-time model needs, as JSON types: the subject prior
    (None when no benchmark was left to fit it) and the hyperparameters. The
    kind tag keeps it apart from the Predictor's prior.json, whose coefficients
    are on the accuracy-logit scale, half as wide."""
    return {"kind": KIND, "prior": None if prior is None else prior.to_dict(),
            "hyper": hyper.to_dict()}


def from_json(d):
    """(SubjectPrior or None, Hyper) from to_json's output. Raises ValueError on
    anything the run-time model could not use: a wrong kind, a malformed
    subject prior, a non-finite or negative hyperparameter."""
    from .hier import Hyper
    if not isinstance(d, dict) or d.get("kind") != KIND:
        raise ValueError(f"not a {KIND} bundle")
    prior = None if d.get("prior") is None else SubjectPrior.from_dict(d["prior"])
    try:
        hyper = Hyper.from_dict(d["hyper"])
    except (TypeError, KeyError, ValueError) as err:
        raise ValueError(f"hyperparameters: {err!r}") from err
    for k, v in hyper.to_dict().items():
        if k not in ("excluded", "included") and \
                (not math.isfinite(v) or (k not in ("mu0", "shift") and v < 0)):
            raise ValueError(f"hyperparameter {k} = {v}")
    return prior, hyper


def save(path, prior, hyper):
    with open(path, "w") as f:
        json.dump(to_json(prior, hyper), f, indent=1)


def load(path):
    with open(path) as f:
        return from_json(json.load(f))


def main():
    """Print the empirical-Bayes hyperparameters on all five public benchmarks
    and leaving each out, with the levels and group shares behind them."""
    from .data import load_pairs
    from .official import eligible
    pairs = eligible(load_pairs())
    names = sorted({p.benchmark_id for p in pairs})
    for held in [()] + [(b,) for b in names]:
        prior = build_prior(pairs, held)
        hyper, rep = fit_hyper(pairs, held, prior)
        h = {k: round(v, 3) for k, v in hyper.to_dict().items() if k not in ("excluded", "included")}
        print(f"excluding {list(held) or 'nothing'}: {h}")
        shares = [round(x, 3) for x in rep.get("shares", [])]
        print(f"   levels at attribute 0 {({b: round(v, 2) for b, v in rep['levels'].items()})} "
              f"(fitted {({b: round(v, 2) for b, v in rep['levels_fitted'].items()})}); "
              f"item variance {rep.get('item_var', float('nan')):.2f}, group shares {shares}; "
              f"link weight {hyper.link_weight:.3f}; fallback {rep['fallback'] or 'none'}"
              + (f"; approximate {rep['approximate']}" if rep["approximate"] else ""))


if __name__ == "__main__":
    main()
