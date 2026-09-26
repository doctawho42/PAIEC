"""Step 2: how should a subject's ability prior be built so it transfers to an
unseen benchmark?

Target: a pair's relative standing z = smoothed logit of its accuracy minus its
benchmark's mean (paiec.subjects.subject_frame), and the same on the Rasch scale
(per-benchmark joint fit, theta centred within the benchmark), because the
Predictor's `a` sits on the first scale while the IRT is off and on the second
once it is on. Leave-one-benchmark-out over the four benchmarks with more than
one subject; swe_rebench (one subject, z = 0 by construction) is left out
everywhere.

    python identity_prior.py > identity_prior_out.txt   # ~2.5 minutes

The nested shrink slope needs two inner training benchmarks; with only three
benchmarks in all (the step-2 code_only.py, not kept) it is 0 and mse_shrunk
equals mse0.
"""
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paiec import official as O  # noqa: E402
from paiec.data import SUBJECT_FIELDS, load_pairs
from paiec.rasch import rasch
from paiec.subjects import Spec, design_matrix, subject_frame

HELD = ["matharena", "multi_swebench", "real_webagents", "researchcodebench"]
KEYS = ["name", "name_effort", "full", "canon"]
PROVIDERS = {"alibaba", "amazon", "anthropic", "bytedance", "cohere", "deepseek",
             "google", "meta", "minimax", "mistral", "moonshot", "nvidia", "openai",
             "stanford", "stepfun", "xai", "zhipu"}
pd.set_option("display.width", 250)


# --- keys --------------------------------------------------------------------

def canon_name(s: dict) -> str:
    """normalized_name, else the source_model_name inside subject_features_extra
    (47% of matharena subjects have an empty name), lower-cased, parentheticals
    such as '(Think)' or '(high)' dropped, provider words dropped."""
    name = s.get("normalized_name", "").strip()
    if not name:
        m = re.search(r"source_model_name=([^;]+)", s.get("subject_features_extra", ""))
        name = m[1] if m else ""
    t = re.sub(r"\(.*?\)", " ", name.lower())
    t = re.sub(r"[-_/]", " ", t)
    toks = [w for w in t.split() if w not in PROVIDERS]
    return " ".join(toks)


def keys_of(s: dict) -> dict:
    n = s.get("normalized_name", "").strip()
    return dict(
        name=n or None,
        name_effort=(n + "|" + s.get("reasoning_effort", "").strip()) if n else None,
        full="|".join(s.get(f, "") for f in SUBJECT_FIELDS),
        canon=canon_name(s) or None)


# --- data --------------------------------------------------------------------

def rasch_theta(pairs):
    """Per pair theta from a Rasch fit on every response of its benchmark,
    centred within the benchmark; items linked across subjects, so matharena's
    different competition subsets stop masquerading as ability."""
    out = {}
    by_b = defaultdict(list)
    for i, p in enumerate(pairs):
        by_b[p.benchmark_id].append((i, p))
    for bid, ps in by_b.items():
        if len(ps) < 2:
            continue
        kidx, si, ji, y = {}, [], [], []
        for j, (_, p) in enumerate(ps):
            for r in p.responses:
                ji.append(kidx.setdefault(r.item_key, len(kidx)))
                si.append(j); y.append(r.label)
        th, _ = rasch(np.array(si), np.array(ji), np.array(y, float), len(ps), len(kidx),
                      lam_t=0.01, lam_b=1.0)
        th = th - th.mean()
        for j, (i, _) in enumerate(ps):
            out[i] = th[j]
    return out


def frame():
    pairs = O.eligible(load_pairs())
    df = subject_frame(pairs)
    for f in SUBJECT_FIELDS:                 # raw fields, prefixed: attrs() owns
        df["f_" + f] = [p.subject.get(f, "") for p in pairs]   # provider/harness
    for k in KEYS:
        df[k] = [keys_of(p.subject)[k] for p in pairs]
    th = rasch_theta(pairs)
    df["theta"] = [th.get(i, 0.0) for i in range(len(pairs))]
    # sampling variance of the accuracy logit (delta method, smoothed)
    df["noise"] = 1.0 / ((df.acc * df.n + 0.5) * (1 - df.acc + 0.5 / df.n))
    return df[df.benchmark_id.isin(HELD)].reset_index(drop=True)


# --- variance components -------------------------------------------------------

def cells(df, key, col):
    """Per (key, benchmark) mean of col over a key's variants in the benchmark
    (multi_swebench harnesses, matharena efforts/checkpoints)."""
    d = df[df[key].notna()]
    return d.groupby([key, "benchmark_id"])[col].agg(["mean", "size"]).reset_index()


def offsets(c, key):
    """Benchmark offsets c_b from y_kb = alpha_k + c_b on keys seen on >= 2
    benchmarks (sum c_b = 0): pool composition differs between benchmarks
    (matharena's pool is 2026-heavy), so the same model's centred standing
    shifts with the pool, not with the model."""
    multi = c.groupby(key).benchmark_id.transform("nunique") >= 2
    m = c[multi].copy()
    bs = sorted(m.benchmark_id.unique())
    if len(bs) < 2:
        return {}
    cb = dict.fromkeys(bs, 0.0)
    for _ in range(200):
        m["alpha"] = (m["mean"] - m.benchmark_id.map(cb)).groupby(m[key]).transform("mean")
        new = (m["mean"] - m.alpha).groupby(m.benchmark_id).mean()
        new -= new.mean()
        if max(abs(new[b] - cb[b]) for b in bs) < 1e-10:
            cb = new.to_dict(); break
        cb = new.to_dict()
    return cb


def components(df, key, col, use_offsets=True):
    """tau2 = var(general ability): mean cross-product of a key's standings on two
    different benchmarks; s2d = var(subject x benchmark): within-key variance
    across benchmarks. Both on per-(key, benchmark) means."""
    c = cells(df, key, col)
    cb = offsets(c, key) if use_offsets else {}
    c["u"] = c["mean"] - c.benchmark_id.map(cb).fillna(0.0)
    g = c.groupby(key)
    multi = c[g.benchmark_id.transform("nunique") >= 2]
    if multi.empty:
        return dict(tau2=np.nan, s2d=np.nan, keys=0, cells=0, offsets=cb)
    mu = multi.u.mean()
    xp, wv, dfw = [], 0.0, 0
    for _, grp in multi.groupby(key):
        u = grp.u.values
        for a, b in combinations(range(len(u)), 2):
            xp.append((u[a] - mu) * (u[b] - mu))
        wv += ((u - u.mean()) ** 2).sum(); dfw += len(u) - 1
    return dict(tau2=float(np.mean(xp)) if xp else np.nan,
                s2d=wv / dfw if dfw else np.nan,
                keys=int(multi[key].nunique()), cells=len(multi), offsets=cb)


def pairwise(df, key, col):
    c = cells(df, key, col)
    w = c.pivot(index=key, columns="benchmark_id", values="mean")
    rows = []
    for a, b in combinations(HELD, 2):
        if a not in w or b not in w:
            continue
        x = w[[a, b]].dropna()
        if len(x) >= 3:
            rows.append(dict(pair=f"{a} / {b}", n=len(x),
                             pearson=pearsonr(x[a], x[b])[0],
                             spearman=spearmanr(x[a], x[b])[0]))
    return pd.DataFrame(rows)


def variants(df, key, col):
    """Variance among a key's variants inside one benchmark (harnesses on
    multi_swebench, efforts and checkpoints on matharena), pooled with df."""
    d = df[df[key].notna()].copy()
    g = d.groupby([key, "benchmark_id"])[col]
    d["dev"] = d[col] - g.transform("mean")
    d["size"] = g.transform("size")
    d = d[d["size"] >= 2]
    ncell = d.groupby([key, "benchmark_id"]).ngroups
    by = {b: round(float((x.dev ** 2).sum() / (len(x) - x.groupby(key).ngroups)), 3)
          for b, x in d.groupby("benchmark_id")}
    return dict(pooled=float((d.dev ** 2).sum() / (len(d) - ncell)), cells=ncell,
                pairs=len(d), by=by)


def mixed(df, key, col):
    """REML check: col ~ C(benchmark) + (1 | key) + (1 | key:benchmark) + residual
    (variants within a key-benchmark cell)."""
    import statsmodels.formula.api as smf
    d = df[df[key].notna()].copy()
    d["k"] = d[key]
    md = smf.mixedlm(f"{col} ~ C(benchmark_id)", d, groups="k", re_formula="1",
                     vc_formula={"kb": "0 + C(benchmark_id)"})
    r = md.fit(reml=True, method="lbfgs")
    return dict(tau2=float(r.cov_re.iloc[0, 0]), s2d=float(r.vcomp[0]), s2v=float(r.scale))


# --- prior constructions -------------------------------------------------------

def small_design(d, med_days, med_ls):
    """The attribute design without the provider and effort dummies."""
    return np.column_stack([
        np.ones(len(d)), (d.days.fillna(med_days).values - 400) / 400,
        d.has_date.values.astype(float), d.log_size.fillna(med_ls).values,
        d.has_size.values.astype(float), d.small.values.astype(float),
        d.big.values.astype(float), d.think.values.astype(float),
        d.harness.values.astype(float)])


def fill(tr, te, key="canon"):
    """Missing release date and provider completed from the same canonical
    model elsewhere in the training table (identity used to complete the
    attributes rather than as a standing)."""
    days = tr[tr.has_date == 1].groupby(key).days.median()
    prov = tr[tr.provider != "unknown"].groupby(key).provider.agg(lambda s: s.mode()[0])

    def f(d):
        d = d.copy()
        m = (d.has_date == 0) & d[key].isin(days.index)
        d.loc[m, "days"] = d.loc[m, key].map(days)
        d.loc[m, "has_date"] = 1
        m = (d.provider == "unknown") & d[key].isin(prov.index)
        d.loc[m, "provider"] = d.loc[m, key].map(prov)
        return d
    return f(tr), f(te)


def attr_fit(tr, col, spec_str="attr"):
    """spec_str: 'attr' (the paiec.subjects design, as fit_prior), 'date' (release
    date alone), with '+small' (no provider/effort dummies), '+fe' (benchmark
    intercepts, nearly unpenalised, dropped at prediction: pools differ between
    benchmarks, so the centred target mixes pool composition into the slopes),
    '+fill' (missing date/provider from the identity table)."""
    parts = spec_str.split("+")
    base, opts = parts[0], set(parts[1:])
    if "fill" in opts:
        tr0 = tr
        tr, _ = fill(tr0, tr0)
    med_d, med_l = tr.days.median(), tr.log_size.median()
    if base == "date":
        X = lambda d: np.column_stack([np.ones(len(d)), (d.days.fillna(med_d).values - 400) / 400,
                                       d.has_date.values.astype(float)])
    elif "small" in opts:
        X = lambda d: small_design(d, med_d, med_l)
    else:
        spec = Spec.from_frame(tr)
        X = lambda d: design_matrix(d, spec)
    Xtr = X(tr)
    k = Xtr.shape[1]
    if "fe" in opts:
        bs = sorted(tr.benchmark_id.unique())
        Xtr = np.hstack([Xtr, np.column_stack([(tr.benchmark_id == b).values * 30.0 for b in bs])])
    coef = Ridge(alpha=2.0, fit_intercept=False).fit(Xtr, tr[col].values).coef_[:k]

    def predict(te):
        if "fill" in opts:
            _, te = fill(tr0, te)
        return X(te) @ coef
    return predict


def identity_obs(tr, te, key, col, resid=None, use_offsets=True):
    """For each test row: (mean over training benchmarks of the key's standing,
    number of those benchmarks). `resid` replaces col by an attribute residual."""
    d = tr.assign(v=tr[col] if resid is None else resid)
    c = cells(d, key, "v")
    if use_offsets:
        cb = offsets(c, key)
        c["mean"] = c["mean"] - c.benchmark_id.map(cb).fillna(0.0)
    g = c.groupby(key)["mean"].agg(["mean", "size"])
    ybar = te[key].map(g["mean"]).values.astype(float)
    k = te[key].map(g["size"]).fillna(0).values.astype(float)
    return ybar, k


def attr_oob(tr, col, spec_str="attr"):
    """Attribute predictions for training rows, each benchmark predicted by a
    ridge fitted without it: residuals a new benchmark would see."""
    out = pd.Series(np.nan, index=tr.index)
    for b in tr.benchmark_id.unique():
        inner, t = tr[tr.benchmark_id != b], tr[tr.benchmark_id == b]
        p = attr_fit(inner, col, spec_str)(t)
        out[tr.benchmark_id == b] = p - p.mean()      # level-free, as the target
    return out.values


def method(name, tr, te, col, key="canon"):
    """Prior mean for te's rows from tr only. Returns (prediction, covered).

    zero | attr[+small][+fe][+fill] | date[+fe] | id_raw | id_eb |
    id_attr:<attr spec> (precision-weighted: attribute mean plus the shrunk
    identity residual) | stack:<attr spec> (weights on attribute and identity
    fitted on the training benchmarks' own leave-one-out predictions)."""
    ones = np.ones(len(te), bool)
    if name == "zero":
        return np.zeros(len(te)), ones
    if name.startswith("attr") or name.startswith("date"):
        return attr_fit(tr, col, name)(te), ones
    if name in ("id_raw", "id_eb"):
        ybar, k = identity_obs(tr, te, key, col)
        cov = k > 0
        if name == "id_raw":
            return np.where(cov, ybar, 0.0), cov
        vc = components(tr, key, col)
        tau2, s2d = np.nan_to_num(vc["tau2"], nan=0.3), np.nan_to_num(vc["s2d"], nan=0.6)
        tau2, s2d = max(tau2, 0.05), max(s2d, 0.05)
        w = np.where(cov, tau2 / (tau2 + s2d / np.maximum(k, 1)), 0.0)
        return np.where(cov, w * ybar, 0.0), cov
    if name.startswith("id_attr:"):
        spec_str = name.split(":", 1)[1]
        m = attr_fit(tr, col, spec_str)(te)
        r = tr[col].values - tr.groupby("benchmark_id")[col].transform("mean").values \
            - attr_oob(tr, col, spec_str)
        rbar, k = identity_obs(tr, te, key, col, resid=r)
        cov = k > 0
        vc = components(tr.assign(r=r), key, "r")
        tau2, s2d = np.nan_to_num(vc["tau2"], nan=0.02), np.nan_to_num(vc["s2d"], nan=0.6)
        tau2, s2d = max(tau2, 0.02), max(s2d, 0.05)
        w = np.where(cov, tau2 / (tau2 + s2d / np.maximum(k, 1)), 0.0)
        return m + np.where(cov, w * rbar, 0.0), cov
    if name.startswith("stack:"):
        spec_str = name.split(":", 1)[1]
        F, Z = [], []
        for b in tr.benchmark_id.unique():
            inner, t = tr[tr.benchmark_id != b], tr[tr.benchmark_id == b]
            if inner.benchmark_id.nunique() < 2:
                continue
            pa = attr_fit(inner, col, spec_str)(t)
            yb, kk = identity_obs(inner, t, key, col)
            pi = np.where(kk > 0, yb, 0.0)
            F.append(np.column_stack([pa - pa.mean(), pi - pi.mean()]))
            Z.append(t[col].values - t[col].values.mean())
        w = (np.linalg.lstsq(np.vstack(F), np.concatenate(Z), rcond=None)[0]
             if F else np.array([1.0, 0.0]))
        pa = attr_fit(tr, col, spec_str)(te)
        yb, kk = identity_obs(tr, te, key, col)
        return w[0] * pa + w[1] * np.where(kk > 0, yb, 0.0), kk > 0
    raise ValueError(name)


def metrics(p, z):
    pc = p - p.mean()
    out = dict(n=len(z), mse0=float(np.mean(z ** 2)),
               mse=float(np.mean((p - z) ** 2)), mse_c=float(np.mean((pc - z) ** 2)))
    if np.std(p) > 1e-9 and len(z) > 2:
        out.update(pearson=pearsonr(p, z)[0], spearman=spearmanr(p, z)[0])
    else:
        out.update(pearson=np.nan, spearman=np.nan)
    return out


def slope(name, tr, col, key):
    """No-intercept shrink factor on centred predictions, fitted on the training
    benchmarks' own leave-one-out predictions (nested, so honest)."""
    num = den = 0.0
    for b in tr.benchmark_id.unique():
        inner, t = tr[tr.benchmark_id != b], tr[tr.benchmark_id == b]
        if inner.benchmark_id.nunique() < 2:
            continue
        p, _ = method(name, inner, t, col, key)
        pc = p - p.mean(); zc = t[col].values - t[col].values.mean()
        num += pc @ zc; den += pc @ pc
    return num / den if den > 0 else 0.0


def lobo(df, col, names, key="canon"):
    rows = []
    for b in HELD:
        tr, te = df[df.benchmark_id != b], df[df.benchmark_id == b]
        z = te[col].values - te[col].values.mean()
        for nm in names:
            p, cov = method(nm, tr, te, col, key)
            s = slope(nm, tr, col, key)
            r = metrics(p, z)
            pc = p - p.mean()
            cov = te[key].isin(set(tr[key].dropna())).values   # same subset for every method
            r.update(benchmark=b, method=nm, slope=s,
                     mse_shrunk=float(np.mean((s * pc - z) ** 2)), coverage=cov.mean())
            if cov.any() and cov.sum() > 2:
                pcv, zcv = p[cov], te[col].values[cov]
                zcv = zcv - zcv.mean()
                rc = metrics(pcv, zcv)
                r.update(cov_n=int(cov.sum()), cov_pearson=rc["pearson"],
                         cov_mse_c=rc["mse_c"], cov_mse0=rc["mse0"])
            rows.append(r)
    return pd.DataFrame(rows)


def pooled(res):
    """Across the four held-out benchmarks, row-weighted."""
    g = res.groupby("method")
    f = lambda c: g.apply(lambda d: np.average(d[c], weights=d.n))
    fc = lambda c: g.apply(lambda d: np.average(d[c].fillna(0), weights=d.cov_n.fillna(0))
                           if d.cov_n.fillna(0).sum() else np.nan)
    return pd.DataFrame(dict(mse0=f("mse0"), mse=f("mse"), mse_c=f("mse_c"),
                             mse_shrunk=f("mse_shrunk"),
                             mean_pearson=g.pearson.mean(), mean_spearman=g.spearman.mean(),
                             cov_mse0=fc("cov_mse0"), cov_mse_c=fc("cov_mse_c"),
                             cov_mean_pearson=g.cov_pearson.mean()))


# --- uncertainty ---------------------------------------------------------------

def boot_components(df, key, col, reps=400, seed=0):
    """Key-cluster bootstrap of (tau2, s2d): resample linked keys with all their
    cells; offsets re-estimated each time."""
    c = cells(df, key, col)
    linked = c[c.groupby(key).benchmark_id.transform("nunique") >= 2]
    ks = linked[key].unique()
    by = {k: g for k, g in linked.groupby(key)}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        pick = rng.choice(ks, len(ks))
        d = pd.concat([by[k].assign(**{key: f"{k}#{i}"}) for i, k in enumerate(pick)])
        d = d.rename(columns={"mean": col})
        vc = components(d.assign(n=1), key, col)
        out.append((vc["tau2"], vc["s2d"]))
    a = np.array(out)
    return np.nanpercentile(a, [2.5, 97.5], axis=0)


def resid(df, col, spec_str="attr"):
    """Leave-one-benchmark-out attribute residual, centred within benchmark."""
    r = np.full(len(df), np.nan)
    for b in HELD:
        m = (df.benchmark_id == b).values
        p = attr_fit(df[~m], col, spec_str)(df[m])
        zc = df[col].values[m] - df[col].values[m].mean()
        r[m] = zc - (p - p.mean())
    return r


# --- main ----------------------------------------------------------------------

ATTRS = ["attr", "attr+fe", "attr+small", "attr+small+fe", "attr+fill", "attr+fill+fe",
         "date", "date+fe"]


def report_decomposition(df, col):
    tot = np.average(df.groupby("benchmark_id")[col].var(), weights=df.groupby("benchmark_id").size())
    print(f"  total within-benchmark var of {col}: {tot:.3f}")
    for k in ["name", "canon"]:
        for off in (True, False):
            vc = components(df, k, col, use_offsets=off)
            print(f"  key={k:<6} offsets={off!s:<5} tau2={vc['tau2']:.3f} s2d={vc['s2d']:.3f} "
                  f"rho={vc['tau2'] / (vc['tau2'] + vc['s2d']):.2f} "
                  f"keys={vc['keys']} cells={vc['cells']} "
                  f"offsets={ {b: round(v, 2) for b, v in vc['offsets'].items()} }")
        lo, hi = boot_components(df, k, col)
        print(f"  key={k:<6} bootstrap 95%: tau2 [{lo[0]:.3f}, {hi[0]:.3f}]  s2d [{lo[1]:.3f}, {hi[1]:.3f}]")
        vv = variants(df, k, col)
        print(f"  key={k:<6} variants within a key-benchmark cell: pooled var={vv['pooled']:.3f} "
              f"({vv['cells']} cells, {vv['pairs']} pairs); by benchmark {vv['by']}")
        try:
            mx = mixed(df, k, col)
            print(f"  key={k:<6} REML (benchmark fixed effects from all rows): tau2={mx['tau2']:.3f} "
                  f"s2d={mx['s2d']:.3f} s2v={mx['s2v']:.3f}")
        except Exception as e:                           # pragma: no cover
            print("  REML failed:", e)
    for spec_str in ["attr", "attr+fe+fill", "date+fe"]:
        r = resid(df, col, spec_str)
        d = df.assign(r=r)
        vc = components(d, "canon", "r")
        lo, hi = boot_components(d, "canon", "r")
        print(f"  {spec_str:<13} LOBO residual var={np.mean(r ** 2):.3f} ({np.mean(r ** 2) / tot:.2f} of total); "
              f"shared across benchmarks on linked keys: tau2_res={vc['tau2']:.3f} "
              f"[{lo[0]:.3f}, {hi[0]:.3f}], s2d_res={vc['s2d']:.3f}")
        if spec_str == "attr":
            print("   residual correlation across benchmark pairs (canon):")
            print(pairwise(d, "canon", "r").round(3).to_string(index=False))


def main():
    df = frame()
    print(f"{len(df)} pairs on {df.benchmark_id.nunique()} multi-subject benchmarks\n")

    print("== keys: how many pairs each key links to another benchmark ==")
    for k in KEYS:
        nb = df[df[k].notna()].groupby(k).benchmark_id.transform("nunique")
        print(f"  {k:<12} keyed {df[k].notna().mean():.2f}  "
              f"linked {(nb >= 2).sum():>3} pairs, {df.loc[nb[nb >= 2].index, k].nunique():>2} keys")

    print("\n== coverage: share of the held-out benchmark's pairs whose key is on another benchmark ==")
    cov = []
    for b in HELD:
        te, tr = df[df.benchmark_id == b], df[df.benchmark_id != b]
        row = dict(benchmark=b, n=len(te))
        for k in KEYS:
            row[k] = te[k].isin(set(tr[k].dropna())).mean()
        cov.append(row)
    print(pd.DataFrame(cov).round(2).to_string(index=False))

    print("\n== do fields differ for one normalized_name across benchmarks? ==")
    nm = df[df["name"].notna()]
    multi = nm[nm.groupby("name").benchmark_id.transform("nunique") >= 2]
    for f in ["provider", "release_date", "harness", "harness_version",
              "reasoning_effort", "access_date", "subject_features_extra"]:
        vals = multi.groupby("name")["f_" + f].nunique()
        print(f"  {f:<24} takes >1 value for {(vals > 1).sum():>2} of {len(vals)} linked names")
    for f in ["harness", "reasoning_effort", "normalized_name", "release_date"]:
        print(f"  non-empty {f} by benchmark:",
              df.groupby("benchmark_id")["f_" + f].agg(lambda s: round((s != "").mean(), 2)).to_dict())

    ms = df[df.benchmark_id == "multi_swebench"]
    import statsmodels.formula.api as smf
    for f in ["z ~ C(name)", "z ~ C(f_harness)", "z ~ C(name) + C(f_harness)"]:
        r = smf.ols(f, ms).fit()
        print(f"  multi_swebench {f:<28} R2={r.rsquared:.3f} adj={r.rsquared_adj:.3f}")
    ma = df[(df.benchmark_id == "matharena") & (df.f_reasoning_effort != "")]
    ma = ma[ma.canon.duplicated(keep=False)].sort_values(["canon", "f_reasoning_effort"])
    print("  matharena effort variants of one model (z = accuracy scale, theta = Rasch):")
    print(ma[["canon", "f_reasoning_effort", "n", "z", "theta"]].round(2).to_string(index=False))

    for col, label in [("z", "accuracy-logit scale"), ("theta", "Rasch scale")]:
        print(f"\n################ {label} ({col}) ################")
        print("per-benchmark sd of the target:",
              df.groupby("benchmark_id")[col].std().round(3).to_dict())
        if col == "z":
            print("mean sampling variance of z:",
                  df.groupby("benchmark_id").noise.mean().round(4).to_dict())
        print("\n-- between-benchmark consistency of standing (per key-benchmark means) --")
        for k in ["name", "canon"]:
            print(f" key={k}")
            print(pairwise(df, k, col).round(3).to_string(index=False))
        print("\n-- variance decomposition --")
        report_decomposition(df, col)

        for k in KEYS:
            names = (["zero"] + ATTRS + ["id_raw", "id_eb", "id_attr:attr", "id_attr:attr+fe+fill",
                                         "stack:attr+fe+fill"]
                     if k == "canon" else ["id_raw", "id_eb", "id_attr:attr"])
            print(f"\n-- leave-one-benchmark-out, identity key = {k} --")
            res = lobo(df, col, names, key=k)
            show = ["benchmark", "method", "n", "coverage", "pearson", "spearman", "mse0",
                    "mse", "mse_c", "slope", "mse_shrunk", "cov_n", "cov_pearson",
                    "cov_mse0", "cov_mse_c"]
            if k == "canon":
                print(res[[c for c in show if c in res]].round(3).to_string(index=False))
            print(" pooled over held-out benchmarks (mse* row-weighted; cov_* on the pairs "
                  "whose key is covered):")
            print(pooled(res).round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
